"""
train_baseline.py - CE-GAN baseline training script.

Supports two modes:
  baseline  - standard within-dataset CEGAN (Scenario A)
  cross     - cross-dataset training with MMD alignment (Scenario C)

Usage examples
--------------
  # Within-dataset baseline on NSL-KDD (synthetic data if CSV not found)
  python experiments/train_baseline.py --source nsl_kdd --mode baseline

  # Cross-dataset with MMD alignment
  python experiments/train_baseline.py --source nsl_kdd --target unsw_nb15 \\
      --mode cross --gamma 0.05 --epochs 200

  # Override architecture dims
  python experiments/train_baseline.py --source cic_ids2017 --d_model 256 --n_layers 4
"""

import argparse
import sys
from pathlib import Path

# Allow 'from src.X import ...' whether the script is run from project/ or experiments/
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import torch.utils.data as data_utils
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -- Dataset config -------------------------------------------------------------

_DS_CONFIG = {
    "nsl_kdd":     {"n_features": 41,  "n_classes": 5},
    "unsw_nb15":   {"n_features": 49,  "n_classes": 10},
    "cic_ids2017": {"n_features": 78,  "n_classes": 15},
}

# -- Argument parser ------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CE-GAN training (baseline or cross-dataset)"
    )
    p.add_argument(
        "--source", default="nsl_kdd",
        choices=list(_DS_CONFIG),
        help="Source dataset name (default: nsl_kdd)",
    )
    p.add_argument(
        "--target", default=None,
        choices=list(_DS_CONFIG),
        help="Target dataset name (required for --mode cross)",
    )
    p.add_argument(
        "--mode", default="baseline",
        choices=["baseline", "cross"],
        help="Training mode: baseline (within-dataset) or cross (MMD alignment)",
    )
    p.add_argument("--epochs",     type=int,   default=200,  help="Training epochs")
    p.add_argument("--batch_size", type=int,   default=256,  help="Mini-batch size")
    p.add_argument("--gamma",      type=float, default=0.05, help="MMD loss weight (cross mode)")
    p.add_argument("--d_model",    type=int,   default=128,  help="Transformer d_model")
    p.add_argument("--n_layers",   type=int,   default=3,    help="Transformer depth")
    p.add_argument(
        "--synthetic", action="store_true",
        help="Force synthetic data even if CSV files are present",
    )
    p.add_argument(
        "--ckpt_every", type=int, default=50,
        help="Save checkpoint every N epochs (default 50)",
    )
    p.add_argument(
        "--log_every", type=int, default=10,
        help="Print loss every N epochs (default 10)",
    )
    return p.parse_args()


# -- Data loading ---------------------------------------------------------------

def get_data(name: str, force_synthetic: bool = False) -> tuple:
    """
    Try to load a real dataset via DatasetLoader.
    Falls back to synthetic Gaussian data on FileNotFoundError.

    Returns (features_tensor, labels_tensor, n_features, n_classes).
    Both tensors are on CPU; the training loop moves batches to device.
    """
    cfg = _DS_CONFIG[name]
    n_features = cfg["n_features"]
    n_classes  = cfg["n_classes"]

    if not force_synthetic:
        try:
            from src.datasets import DatasetLoader
            loader = DatasetLoader(name)
            loader.load()
            X = loader.features.cpu()
            y = loader.labels.cpu()
            print(f"  Loaded real {name}: {X.shape[0]:,} samples x {n_features} features")
            return X, y, n_features, n_classes
        except Exception as exc:
            print(f"  [{name}] DatasetLoader failed ({exc}); using synthetic data.")

    n_synth = 4000
    X = torch.rand(n_synth, n_features)
    y = torch.randint(0, n_classes, (n_synth,))
    print(f"  Synthetic {name}: {n_synth} samples x {n_features} features")
    return X, y, n_features, n_classes


def make_loader(
    X: torch.Tensor, y: torch.Tensor, batch_size: int
) -> data_utils.DataLoader:
    ds = data_utils.TensorDataset(X, y)
    return data_utils.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)


# -- Loss tracker ---------------------------------------------------------------

class LossTracker:
    """Accumulates per-step losses and returns per-epoch averages."""

    def __init__(self) -> None:
        self._sums: dict = {}
        self._counts: dict = {}

    def update(self, losses: dict) -> None:
        for k, v in losses.items():
            self._sums[k]   = self._sums.get(k, 0.0) + v
            self._counts[k] = self._counts.get(k, 0)  + 1

    def averages(self) -> dict:
        return {k: self._sums[k] / self._counts[k] for k in self._sums}

    def reset(self) -> None:
        self._sums.clear()
        self._counts.clear()


# -- Training loops -------------------------------------------------------------

def _train_epoch_baseline(model, src_loader, tracker) -> None:
    model.train()
    for xb, yb in src_loader:
        losses = model.train_step(xb, yb)
        tracker.update(losses)


def _train_epoch_cross(model, src_loader, tgt_loader, tracker) -> None:
    model.train()
    tgt_iter = iter(tgt_loader)

    for xb, yb in src_loader:
        try:
            xt, _ = next(tgt_iter)
        except StopIteration:
            tgt_iter = iter(tgt_loader)
            xt, _ = next(tgt_iter)

        # Pad/truncate target to source feature dim for latent-space MMD
        n_src, n_tgt = xb.size(1), xt.size(1)
        if n_tgt < n_src:
            xt = torch.cat([xt, torch.zeros(xt.size(0), n_src - n_tgt)], dim=1)
        elif n_tgt > n_src:
            xt = xt[:, :n_src]

        losses = model.train_cross_dataset_step(xb, yb, xt)
        tracker.update(losses)


# -- Loss curve plot ------------------------------------------------------------

def save_loss_curves(history: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = list(range(1, len(next(iter(history.values()))) + 1))

    axes[0].plot(epochs, history.get("d_loss", []), label="D loss")
    axes[0].plot(epochs, history.get("g_loss", []), label="G loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_title("Discriminator vs Generator Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for k in history:
        if k not in ("d_loss", "g_loss"):
            axes[1].plot(epochs, history[k], label=k)
    axes[1].set_xlabel("Epoch")
    axes[1].set_title("Generator Loss Components")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(path.stem.replace("_", " ").title(), fontsize=12)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Loss curves saved -> {path}")


# -- Main -----------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.mode == "cross" and args.target is None:
        print("Error: --target is required for --mode cross")
        sys.exit(1)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device_str}")
    print(f"Mode   : {args.mode}")
    print(f"Source : {args.source}")
    if args.mode == "cross":
        print(f"Target : {args.target}")
    print(f"Epochs : {args.epochs}  |  batch={args.batch_size}")
    print()

    # -- Load data ---------------------------------------------------------------
    print("Loading source data...")
    X_src, y_src, n_feat_src, n_cls_src = get_data(args.source, args.synthetic)
    src_loader = make_loader(X_src, y_src, args.batch_size)

    tgt_loader = None
    if args.mode == "cross":
        print("Loading target data...")
        X_tgt, y_tgt, _, _ = get_data(args.target, args.synthetic)
        tgt_loader = make_loader(X_tgt, y_tgt, args.batch_size)

    # -- Build model -------------------------------------------------------------
    from src.cegan import CEGANConfig, CEGAN, CrossDatasetCEGAN

    cfg = CEGANConfig(
        n_features=n_feat_src,
        n_classes=n_cls_src,
        d_model=args.d_model,
        n_layers=args.n_layers,
        device=device_str,
    )

    model = CrossDatasetCEGAN(cfg, gamma=args.gamma) if args.mode == "cross" else CEGAN(cfg)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel params: {n_params:,}\n")

    # -- Paths -------------------------------------------------------------------
    ckpt_dir = _PROJECT_ROOT / "results" / "checkpoints"
    fig_dir  = _PROJECT_ROOT / "results" / "figures"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.mode}_{args.source}"

    # -- tqdm setup --------------------------------------------------------------
    try:
        from tqdm import tqdm

        def _log(msg: str) -> None:
            tqdm.write(msg)

        def _epoch_bar(n: int):
            return tqdm(range(1, n + 1), desc="Epochs", unit="ep")

    except ImportError:
        def _log(msg: str) -> None:  # type: ignore[misc]
            print(msg)

        def _epoch_bar(n: int):      # type: ignore[misc]
            return range(1, n + 1)

    # -- Training loop -----------------------------------------------------------
    history: dict = {}
    tracker = LossTracker()
    print("Starting training...\n")

    for epoch in _epoch_bar(args.epochs):
        tracker.reset()

        if args.mode == "cross":
            _train_epoch_cross(model, src_loader, tgt_loader, tracker)
        else:
            _train_epoch_baseline(model, src_loader, tracker)

        avgs = tracker.averages()
        for k, v in avgs.items():
            history.setdefault(k, []).append(v)

        if epoch % args.log_every == 0 or epoch == 1:
            parts = [f"epoch={epoch:4d}/{args.epochs}"]
            for k in ("d_loss", "g_loss", "l_mmd"):
                if k in avgs:
                    parts.append(f"{k}={avgs[k]:.4f}")
            _log("  " + "  ".join(parts))

        if epoch % args.ckpt_every == 0:
            ckpt = ckpt_dir / f"{tag}_ep{epoch:04d}.pt"
            model.save_checkpoint(ckpt)
            _log(f"  Checkpoint -> {ckpt.name}")

    # -- Final checkpoint + curves -----------------------------------------------
    final_ckpt = ckpt_dir / f"{tag}_final.pt"
    model.save_checkpoint(final_ckpt)
    print(f"\nFinal checkpoint -> {final_ckpt}")

    save_loss_curves(history, fig_dir / f"training_loss_{tag}.png")
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
