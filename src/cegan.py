from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class CEGANConfig:
    # Dataset dimensions – override per dataset before constructing CEGAN
    n_features: int = 41        # NSL-KDD default; 49 for UNSW-NB15, 78 for CIC-IDS2017
    n_classes: int = 5

    # Transformer hyperparameters
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    d_ff: int = 256             # feedforward dim inside each Transformer block
    d_latent: int = 64          # encoder output / latent space dim
    dropout: float = 0.1

    # Generator noise
    noise_dim: int = 64
    n_noise_tokens: int = 8     # noise is expanded into this many tokens

    # CE-GAN composite loss weights (Yang et al. 2025)
    alpha: float = 0.1          # L_cst (conditional structure) weight
    beta: float = 0.01          # L_mmt (moment matching) weight

    # Optimiser
    lr_G: float = 2e-4
    lr_D: float = 2e-4

    # Training stability
    grad_clip: float = 1.0
    device: str = "cuda"        # "cuda" or "cpu"


# ── Internal factory helpers ───────────────────────────────────────────────────

def _encoder_stack(cfg: CEGANConfig) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=cfg.d_model,
        nhead=cfg.n_heads,
        dim_feedforward=cfg.d_ff,
        dropout=cfg.dropout,
        batch_first=True,
        norm_first=True,        # Pre-LN: more stable than post-LN for deep stacks
    )
    return nn.TransformerEncoder(layer, num_layers=cfg.n_layers)


def _decoder_stack(cfg: CEGANConfig) -> nn.TransformerDecoder:
    layer = nn.TransformerDecoderLayer(
        d_model=cfg.d_model,
        nhead=cfg.n_heads,
        dim_feedforward=cfg.d_ff,
        dropout=cfg.dropout,
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerDecoder(layer, num_layers=cfg.n_layers)


# ── Encoder ────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """
    Conditional encoder: (features, class_label) → latent z.

    Each input feature is projected to d_model as an individual token.
    The class label is embedded and prepended as a CLS-style token.
    The CLS output is projected to d_latent to form z.
    """

    def __init__(self, cfg: CEGANConfig) -> None:
        super().__init__()
        self.feature_proj = nn.Linear(1, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.n_features, cfg.d_model)
        self.label_emb = nn.Embedding(cfg.n_classes, cfg.d_model)
        self.transformer = _encoder_stack(cfg)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_latent)
        self.norm = nn.LayerNorm(cfg.d_latent)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        x : [B, n_features]  min-max normalised features
        y : [B]              integer class indices
        → z : [B, d_latent]
        """
        B, N = x.shape

        # Each scalar feature → d_model vector + positional offset
        tokens = self.feature_proj(x.unsqueeze(-1))                  # [B, N, d_model]
        tokens = tokens + self.pos_emb(torch.arange(N, device=x.device))

        # Prepend class label as CLS token
        cls = self.label_emb(y).unsqueeze(1)                         # [B, 1, d_model]
        seq = torch.cat([cls, tokens], dim=1)                        # [B, N+1, d_model]

        out = self.transformer(seq)                                   # [B, N+1, d_model]
        z = self.norm(self.out_proj(out[:, 0]))                      # CLS → [B, d_latent]
        return z


# ── Decoder ────────────────────────────────────────────────────────────────────

class Decoder(nn.Module):
    """
    Reconstructs features from latent z.

    Learnable position queries (one per output feature) attend to z via
    Transformer cross-attention, then each query is projected to a scalar.
    """

    def __init__(self, cfg: CEGANConfig) -> None:
        super().__init__()
        self.latent_proj = nn.Linear(cfg.d_latent, cfg.d_model)
        # One learnable query per output feature
        self.pos_queries = nn.Parameter(
            torch.randn(1, cfg.n_features, cfg.d_model) * 0.02
        )
        self.transformer = _decoder_stack(cfg)
        self.out_proj = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Linear(cfg.d_model // 2, 1),
            nn.Sigmoid(),   # output matches min-max [0, 1] input scale
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z   : [B, d_latent]
        → x_rec : [B, n_features]
        """
        mem = self.latent_proj(z).unsqueeze(1)                       # [B, 1, d_model]
        tgt = self.pos_queries.expand(z.size(0), -1, -1)             # [B, n_features, d_model]
        out = self.transformer(tgt, mem)                             # [B, n_features, d_model]
        return self.out_proj(out).squeeze(-1)                        # [B, n_features]


# ── Generator ──────────────────────────────────────────────────────────────────

class Generator(nn.Module):
    """
    Conditional generator: (noise, class_label) → synthetic features.

    Noise is expanded into n_noise_tokens, a class label token is prepended,
    and learnable content queries define per-feature output positions.
    The Transformer attends over [label_token | noise_tokens | content_queries];
    the content query outputs are mapped to scalar feature values via Sigmoid.
    """

    def __init__(self, cfg: CEGANConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.label_emb = nn.Embedding(cfg.n_classes, cfg.d_model)
        self.noise_expand = nn.Linear(cfg.noise_dim, cfg.n_noise_tokens * cfg.d_model)
        self.content_queries = nn.Parameter(
            torch.randn(1, cfg.n_features, cfg.d_model) * 0.02
        )
        self.transformer = _encoder_stack(cfg)
        self.out_proj = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Linear(cfg.d_model // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        z : [B, noise_dim]   Gaussian noise
        y : [B]              target class indices
        → x_fake : [B, n_features]
        """
        B = z.size(0)
        noise_seq = self.noise_expand(z).view(
            B, self.cfg.n_noise_tokens, self.cfg.d_model
        )                                                             # [B, T, d_model]
        label_tok = self.label_emb(y).unsqueeze(1)                   # [B, 1, d_model]
        queries = self.content_queries.expand(B, -1, -1)             # [B, N, d_model]

        seq = torch.cat([label_tok, noise_seq, queries], dim=1)      # [B, 1+T+N, d_model]
        out = self.transformer(seq)

        # Content query outputs are at positions [1+T : 1+T+N]
        feature_out = out[:, 1 + self.cfg.n_noise_tokens:]           # [B, N, d_model]
        return self.out_proj(feature_out).squeeze(-1)                 # [B, n_features]


# ── Discriminator ──────────────────────────────────────────────────────────────

class Discriminator(nn.Module):
    """
    Classifies samples as real/fake and verifies class conditioning (ACGAN-style).

    Returns
    -------
    validity    : [B, 1]          sigmoid real/fake score
    cond_logits : [B, n_classes]  raw condition-match logits (for CE loss)
    """

    def __init__(self, cfg: CEGANConfig) -> None:
        super().__init__()
        self.feature_proj = nn.Linear(1, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.n_features, cfg.d_model)
        self.label_emb = nn.Embedding(cfg.n_classes, cfg.d_model)
        self.transformer = _encoder_stack(cfg)

        self.validity_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model // 2, 1),
            nn.Sigmoid(),
        )
        self.cond_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model // 2, cfg.n_classes),
        )

    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x : [B, n_features]
        y : [B]
        → (validity [B, 1], cond_logits [B, n_classes])
        """
        B, N = x.shape
        tokens = self.feature_proj(x.unsqueeze(-1))
        tokens = tokens + self.pos_emb(torch.arange(N, device=x.device))

        cls = self.label_emb(y).unsqueeze(1)
        seq = torch.cat([cls, tokens], dim=1)                        # [B, N+1, d_model]
        out = self.transformer(seq)
        rep = out[:, 0]                                              # CLS representation

        return self.validity_head(rep), self.cond_head(rep)


# ── CompositeLoss ──────────────────────────────────────────────────────────────

class CompositeLoss:
    """
    CE-GAN composite loss (Yang et al. 2025).

    Generator    : L_G = L_adv + alpha * L_cst + beta * L_mmt + L_rec
    Discriminator: L_D = (L_real + L_fake) / 2  + alpha * L_cond
    """

    def __init__(self, alpha: float = 0.1, beta: float = 0.01) -> None:
        self.alpha = alpha
        self.beta = beta

    # ── individual loss components ────────────────────────────────────────────

    @staticmethod
    def adversarial(fake_validity: torch.Tensor) -> torch.Tensor:
        """L_adv: generator fools discriminator (BCE, labelled as real)."""
        return F.binary_cross_entropy(fake_validity, torch.ones_like(fake_validity))

    @staticmethod
    def conditional_structure(
        cond_logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """L_cst: discriminator must classify fakes to the correct class."""
        return F.cross_entropy(cond_logits, labels)

    @staticmethod
    def moment_matching(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        """L_mmt: match first and second moments of fake vs real distributions."""
        mean_diff = (fake.mean(0) - real.mean(0)).pow(2).mean()
        var_diff = (
            fake.var(0, unbiased=False) - real.var(0, unbiased=False)
        ).pow(2).mean()
        return mean_diff + var_diff

    @staticmethod
    def reconstruction(x_rec: torch.Tensor, x_real: torch.Tensor) -> torch.Tensor:
        """L_rec: encoder-decoder round-trip fidelity (MSE)."""
        return F.mse_loss(x_rec, x_real)

    # ── combined objectives ───────────────────────────────────────────────────

    def generator_loss(
        self,
        fake_validity: torch.Tensor,
        fake_cond_logits: torch.Tensor,
        fake_samples: torch.Tensor,
        real_samples: torch.Tensor,
        x_rec: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Returns (total_loss tensor, component breakdown dict for logging).
        L_G = L_adv + alpha * L_cst + beta * L_mmt + L_rec
        """
        l_adv = self.adversarial(fake_validity)
        l_cst = self.conditional_structure(fake_cond_logits, labels)
        l_mmt = self.moment_matching(fake_samples, real_samples)
        l_rec = self.reconstruction(x_rec, real_samples)

        total = l_adv + self.alpha * l_cst + self.beta * l_mmt + l_rec
        return total, {
            "l_adv": l_adv.item(),
            "l_cst": l_cst.item(),
            "l_mmt": l_mmt.item(),
            "l_rec": l_rec.item(),
        }

    def discriminator_loss(
        self,
        real_validity: torch.Tensor,
        fake_validity: torch.Tensor,
        real_cond_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        real_loss = F.binary_cross_entropy(
            real_validity, torch.ones_like(real_validity)
        )
        fake_loss = F.binary_cross_entropy(
            fake_validity, torch.zeros_like(fake_validity)
        )
        cond_loss = F.cross_entropy(real_cond_logits, labels)
        return (real_loss + fake_loss) / 2 + self.alpha * cond_loss


# ── CEGAN ──────────────────────────────────────────────────────────────────────

class CEGAN(nn.Module):
    """
    CE-GAN: Conditional Encoder GAN for network intrusion detection.

    Owns all four Transformer-based components plus their optimisers.
    Implements the conditional aggregation encoder-decoder design from
    Yang et al. 2025 with composite loss and gradient clipping.
    """

    def __init__(self, cfg: CEGANConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(
            "cuda" if (cfg.device == "cuda" and torch.cuda.is_available()) else "cpu"
        )

        self.encoder = Encoder(cfg).to(self.device)
        self.decoder = Decoder(cfg).to(self.device)
        self.generator = Generator(cfg).to(self.device)
        self.discriminator = Discriminator(cfg).to(self.device)
        self.criterion = CompositeLoss(cfg.alpha, cfg.beta)

        # Encoder, decoder, and generator are co-trained as a single G network
        _G_params = (
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.generator.parameters())
        )
        self.opt_G = Adam(_G_params, lr=cfg.lr_G, betas=(0.5, 0.999))
        self.opt_D = Adam(
            self.discriminator.parameters(), lr=cfg.lr_D, betas=(0.5, 0.999)
        )
        self._step: int = 0

    # ── training ──────────────────────────────────────────────────────────────

    def train_step(
        self, batch: torch.Tensor, labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        One alternating GAN training step: D update then G update.

        Parameters
        ----------
        batch  : [B, n_features]  real samples (min-max normalised)
        labels : [B]              integer class indices

        Returns
        -------
        dict with keys: d_loss, g_loss, l_adv, l_cst, l_mmt, l_rec
        """
        x = batch.to(self.device)
        y = labels.to(self.device)
        B = x.size(0)

        # ── Discriminator step ─────────────────────────────────────────────
        self.opt_D.zero_grad()

        # Generate fakes without accumulating G gradients
        with torch.no_grad():
            noise = torch.randn(B, self.cfg.noise_dim, device=self.device)
            x_fake = self.generator(noise, y)

        real_val, real_cond = self.discriminator(x, y)
        fake_val, _ = self.discriminator(x_fake, y)

        d_loss = self.criterion.discriminator_loss(real_val, fake_val, real_cond, y)
        d_loss.backward()
        nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.cfg.grad_clip)
        self.opt_D.step()

        # ── Generator + Encoder/Decoder step ──────────────────────────────
        self.opt_G.zero_grad()

        noise = torch.randn(B, self.cfg.noise_dim, device=self.device)
        x_fake = self.generator(noise, y)
        fake_val, fake_cond = self.discriminator(x_fake, y)

        z = self.encoder(x, y)
        x_rec = self.decoder(z)

        g_loss, breakdown = self.criterion.generator_loss(
            fake_val, fake_cond, x_fake, x, x_rec, y
        )
        g_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.generator.parameters()),
            self.cfg.grad_clip,
        )
        self.opt_G.step()

        self._step += 1
        return {"d_loss": d_loss.item(), "g_loss": g_loss.item(), **breakdown}

    # ── inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(self, n_samples: int, class_label: int) -> torch.Tensor:
        """
        Generate synthetic minority-class samples.

        Parameters
        ----------
        n_samples   : number of synthetic examples to produce
        class_label : integer class index to condition on

        Returns
        -------
        Tensor [n_samples, n_features] with values in [0, 1] (min-max scale)
        """
        was_training = self.generator.training
        self.generator.eval()

        y = torch.full((n_samples,), class_label, dtype=torch.long, device=self.device)
        noise = torch.randn(n_samples, self.cfg.noise_dim, device=self.device)
        samples = self.generator(noise, y)

        self.generator.train(was_training)
        return samples

    # ── persistence ───────────────────────────────────────────────────────────

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "step": self._step,
                "cfg": self.cfg,
                "encoder": self.encoder.state_dict(),
                "decoder": self.decoder.state_dict(),
                "generator": self.generator.state_dict(),
                "discriminator": self.discriminator.state_dict(),
                "opt_G": self.opt_G.state_dict(),
                "opt_D": self.opt_D.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self._step = ckpt["step"]
        self.encoder.load_state_dict(ckpt["encoder"])
        self.decoder.load_state_dict(ckpt["decoder"])
        self.generator.load_state_dict(ckpt["generator"])
        self.discriminator.load_state_dict(ckpt["discriminator"])
        self.opt_G.load_state_dict(ckpt["opt_G"])
        self.opt_D.load_state_dict(ckpt["opt_D"])


# ── CrossDatasetCEGAN ──────────────────────────────────────────────────────────

class CrossDatasetCEGAN(CEGAN):
    """
    Cross-dataset CE-GAN with MMD latent-space alignment.

    Extends CEGAN with an MMD term in the generator loss that encourages
    the latent representations of generated source-domain samples to
    match those of real target-domain samples (Yang et al. 2025, Sec. 3.3):

        L_cross = L_adv + alpha * L_cst + beta * L_mmt + gamma * L_MMD

    Parameters
    ----------
    cfg            : CEGANConfig sized for the SOURCE dataset
    gamma          : weight for the MMD alignment term (default 0.05)
    target_samples : [n, n_features_source] pre-aligned target samples,
                     or None (must call update_target_samples before training)
    """

    def __init__(
        self,
        cfg: CEGANConfig,
        gamma: float = 0.05,
        target_samples: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__(cfg)
        self.gamma = gamma

        # Lazy import: works whether run as __main__ or imported as src.cegan
        try:
            from src.mmd import MMDLoss as _MMDLoss
        except ImportError:
            from mmd import MMDLoss as _MMDLoss  # type: ignore[import]
        self._mmd = _MMDLoss()

        self._target_samples: Optional[torch.Tensor] = (
            target_samples.to(self.device) if target_samples is not None else None
        )

    # ── cross-dataset generator loss ──────────────────────────────────────────

    def cross_loss(
        self,
        fake_validity: torch.Tensor,
        fake_cond_logits: torch.Tensor,
        generated: torch.Tensor,
        source_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        L_cross = L_adv + alpha * L_cst + beta * L_mmt + gamma * L_MMD

        MMD is computed in latent space: the encoder maps generated source
        samples and pre-aligned target samples to z-vectors, then penalises
        the kernel MMD² distance between the two z-distributions.

        Parameters
        ----------
        fake_validity    : [B, 1]           D output for generated samples
        fake_cond_logits : [B, n_classes]   condition logits from D
        generated        : [B, n_features]  generator output
        source_labels    : [B]              class labels used to condition G

        Returns
        -------
        (total_loss tensor, breakdown dict with float values per component)
        """
        if self._target_samples is None:
            raise RuntimeError(
                "CrossDatasetCEGAN: call update_target_samples() before training."
            )

        l_adv = CompositeLoss.adversarial(fake_validity)
        l_cst = CompositeLoss.conditional_structure(fake_cond_logits, source_labels)
        l_mmt = CompositeLoss.moment_matching(generated, self._target_samples)

        # Latent-space MMD: encode both domains with the shared encoder
        z_src = self.encoder(generated, source_labels)
        tgt_y = torch.zeros(
            self._target_samples.size(0), dtype=torch.long, device=self.device
        )
        z_tgt = self.encoder(self._target_samples, tgt_y)
        l_mmd = self._mmd(z_src, z_tgt)

        total = (
            l_adv
            + self.cfg.alpha * l_cst
            + self.cfg.beta * l_mmt
            + self.gamma * l_mmd
        )
        return total, {
            "l_adv": l_adv.item(),
            "l_cst": l_cst.item(),
            "l_mmt": l_mmt.item(),
            "l_mmd": l_mmd.item(),
        }

    # ── cross-dataset training step ───────────────────────────────────────────

    def train_cross_dataset_step(
        self,
        source_batch: torch.Tensor,
        source_labels: torch.Tensor,
        target_batch: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Alternating GAN update with MMD cross-dataset alignment.

        D step is identical to CEGAN.train_step.
        G step replaces generator_loss() with cross_loss().

        Parameters
        ----------
        source_batch  : [B, n_features]   real source-domain samples
        source_labels : [B]               source class indices
        target_batch  : [Bt, n_features]  pre-aligned target-domain samples
                        (replaces cached _target_samples for this step)

        Returns
        -------
        dict with keys: d_loss, g_loss, l_adv, l_cst, l_mmt, l_mmd
        """
        x = source_batch.to(self.device)
        y = source_labels.to(self.device)
        self.update_target_samples(target_batch)
        B = x.size(0)

        # ── Discriminator step ─────────────────────────────────────────────
        self.opt_D.zero_grad()

        with torch.no_grad():
            noise = torch.randn(B, self.cfg.noise_dim, device=self.device)
            x_fake = self.generator(noise, y)

        real_val, real_cond = self.discriminator(x, y)
        fake_val, _ = self.discriminator(x_fake, y)

        d_loss = self.criterion.discriminator_loss(real_val, fake_val, real_cond, y)
        d_loss.backward()
        nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.cfg.grad_clip)
        self.opt_D.step()

        # ── Generator step with MMD term ───────────────────────────────────
        self.opt_G.zero_grad()

        noise = torch.randn(B, self.cfg.noise_dim, device=self.device)
        x_fake = self.generator(noise, y)
        fake_val, fake_cond = self.discriminator(x_fake, y)

        g_loss, breakdown = self.cross_loss(fake_val, fake_cond, x_fake, y)
        g_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.generator.parameters()),
            self.cfg.grad_clip,
        )
        self.opt_G.step()

        self._step += 1
        return {"d_loss": d_loss.item(), "g_loss": g_loss.item(), **breakdown}

    # ── helpers ───────────────────────────────────────────────────────────────

    def update_target_samples(self, target_samples: torch.Tensor) -> None:
        """Replace cached target samples (called each mini-batch with aligned target data)."""
        self._target_samples = target_samples.to(self.device)

    # ── persistence ───────────────────────────────────────────────────────────

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "step": self._step,
                "cfg": self.cfg,
                "gamma": self.gamma,
                "encoder": self.encoder.state_dict(),
                "decoder": self.decoder.state_dict(),
                "generator": self.generator.state_dict(),
                "discriminator": self.discriminator.state_dict(),
                "opt_G": self.opt_G.state_dict(),
                "opt_D": self.opt_D.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self._step = ckpt["step"]
        self.gamma = ckpt.get("gamma", self.gamma)
        self.encoder.load_state_dict(ckpt["encoder"])
        self.decoder.load_state_dict(ckpt["decoder"])
        self.generator.load_state_dict(ckpt["generator"])
        self.discriminator.load_state_dict(ckpt["discriminator"])
        self.opt_G.load_state_dict(ckpt["opt_G"])
        self.opt_D.load_state_dict(ckpt["opt_D"])


# ── Smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device_str}\n")

    DATASETS = [
        ("NSL-KDD",     41,  5),
        ("UNSW-NB15",   49, 10),
        ("CIC-IDS2017", 78, 15),
    ]

    # ── Test 1: CEGAN within-dataset ──────────────────────────────────────────
    print("=== CEGAN within-dataset ===")
    for ds_name, n_feat, n_cls in DATASETS:
        cfg = CEGANConfig(n_features=n_feat, n_classes=n_cls, device=device_str)
        model = CEGAN(cfg)

        B = 16
        x = torch.rand(B, n_feat, device=model.device)
        y = torch.randint(0, n_cls, (B,), device=model.device)

        losses = model.train_step(x, y)
        synth = model.generate(n_samples=32, class_label=0)

        assert synth.shape == (32, n_feat), f"Unexpected shape: {synth.shape}"
        assert synth.device.type == model.device.type

        print(
            f"{ds_name:<14}  "
            f"d_loss={losses['d_loss']:.4f}  "
            f"g_loss={losses['g_loss']:.4f}  "
            f"l_adv={losses['l_adv']:.4f}  "
            f"l_rec={losses['l_rec']:.4f}  "
            f"gen={tuple(synth.shape)}"
        )

    # ── Test 2: CrossDatasetCEGAN gradient flow ───────────────────────────────
    print("\n=== CrossDatasetCEGAN gradient flow (NSL-KDD -> UNSW-NB15 space) ===")
    torch.manual_seed(42)
    B = 16
    # Source: NSL-KDD feature space (41 features, 5 classes)
    cfg_src = CEGANConfig(n_features=41, n_classes=5, device=device_str)
    cross_model = CrossDatasetCEGAN(cfg_src, gamma=0.05)

    src_x = torch.rand(B, 41, device=cross_model.device)
    src_y = torch.randint(0, 5, (B,), device=cross_model.device)
    # Target batch is pre-aligned to source (41-dim) feature space
    tgt_x = torch.rand(B, 41, device=cross_model.device)

    losses = cross_model.train_cross_dataset_step(src_x, src_y, tgt_x)

    # Verify all loss components are present and finite
    for key in ("d_loss", "g_loss", "l_adv", "l_cst", "l_mmt", "l_mmd"):
        assert key in losses, f"Missing key: {key}"
        assert torch.isfinite(torch.tensor(losses[key])), f"Non-finite {key}: {losses[key]}"

    print(
        f"{'cross-NSL-KDD':<14}  "
        f"d_loss={losses['d_loss']:.4f}  "
        f"g_loss={losses['g_loss']:.4f}  "
        f"l_mmd={losses['l_mmd']:.4f}"
    )
    print("\nAll smoke tests passed.")
