"""
MMD (Maximum Mean Discrepancy) module for cross-dataset feature alignment.

Implements the biased empirical MMD² estimator with a Gaussian RBF kernel,
as used in the cross-dataset CE-GAN framework (Yang et al. 2025, Eq. 2):

    MMD²(P,Q) = (1/ns²) ΣΣ k(xs,xs)
              − (2/ns·nt) ΣΣ k(xs,xt)
              + (1/nt²) ΣΣ k(xt,xt)

All computation runs on the device of the input tensors (CPU or CUDA).
"""

from pathlib import Path
from typing import Callable, Dict, Optional

import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

KernelFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


# -- Pairwise squared-distance helper ------------------------------------------

def _pairwise_sq_dists(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Efficient pairwise squared Euclidean distances via the identity
        ||x-y||² = ||x||² + ||y||² - 2⟨x,y⟩

    X : [n, d]  Y : [m, d]
    -> [n, m]  (clamped to ≥ 0 to guard against floating-point negatives)
    """
    XX = (X * X).sum(-1, keepdim=True)    # [n, 1]
    YY = (Y * Y).sum(-1, keepdim=True)    # [m, 1]
    XY = X @ Y.T                           # [n, m]
    return (XX + YY.T - 2.0 * XY).clamp(min=0.0)


# -- Median bandwidth heuristic -------------------------------------------------

def median_bandwidth(
    X: torch.Tensor,
    Y: Optional[torch.Tensor] = None,
) -> float:
    """
    Bandwidth selection via the median heuristic.

    σ is set to the median of all pairwise distances in the pooled sample
    {X ∪ Y}.  Only the upper triangle is used (excludes zero self-distances).
    Subsampled to ≤ 2 000 points to keep the O(n²) cost tractable; the
    computation is fully detached so σ never enters the gradient graph.

    Returns
    -------
    float  –  σ > 0 (falls back to 1.0 if median is zero)
    """
    pool = torch.cat([X, Y], dim=0) if Y is not None else X
    n = pool.size(0)

    if n > 2000:
        idx = torch.randperm(n, device=pool.device)[:2000]
        pool = pool[idx]
        n = 2000

    dists_sq = _pairwise_sq_dists(pool.detach(), pool.detach())  # [n, n]
    mask = torch.triu(
        torch.ones(n, n, dtype=torch.bool, device=pool.device), diagonal=1
    )
    upper_dists = dists_sq[mask].sqrt()
    if upper_dists.numel() == 0:
        return 1.0
    sigma = float(upper_dists.median().item())
    return sigma if sigma > 0.0 else 1.0


# -- Gaussian RBF kernel --------------------------------------------------------

def gaussian_rbf_kernel(
    X: torch.Tensor,
    Y: torch.Tensor,
    sigma: Optional[float] = None,
) -> torch.Tensor:
    """
    Gaussian RBF (squared-exponential) kernel matrix.

        K[i, j] = exp(−‖xi − yj‖² / (2σ²))

    If sigma is None the bandwidth is chosen automatically via the
    median heuristic on the pooled samples X and Y.

    Parameters
    ----------
    X     : [n, d]   source samples  (CUDA or CPU tensor)
    Y     : [m, d]   target samples  (same device as X)
    sigma : float | None

    Returns
    -------
    K : [n, m]  kernel matrix  (differentiable w.r.t. X and Y)
    """
    if sigma is None:
        sigma = median_bandwidth(X, Y)

    dists_sq = _pairwise_sq_dists(X, Y)
    return torch.exp(-dists_sq / (2.0 * sigma ** 2))


# -- Biased empirical MMD² estimator -------------------------------------------

def compute_mmd_squared(
    X: torch.Tensor,
    Y: torch.Tensor,
    kernel_fn: Optional[KernelFn] = None,
) -> torch.Tensor:
    """
    Biased empirical MMD² estimator (Yang et al. 2025, Eq. 2):

        MMD²̂ = (1/ns²) ΣΣ k(xs,xs)
               − (2/ns·nt) ΣΣ k(xs,xt)
               + (1/nt²) ΣΣ k(xt,xt)

    When kernel_fn is None, the bandwidth is derived from the pooled
    samples X ∪ Y once and reused across all three kernel evaluations,
    ensuring a consistent scale.

    Parameters
    ----------
    X         : [ns, d]  source samples  (CUDA tensor)
    Y         : [nt, d]  target samples  (CUDA tensor)
    kernel_fn : (A, B) -> K | None  – defaults to Gaussian RBF

    Returns
    -------
    Scalar tensor  –  differentiable w.r.t. X and Y
    """
    if kernel_fn is None:
        # Compute σ once from pooled data so all three matrices share one scale
        sigma = median_bandwidth(X, Y)
        kernel_fn = lambda A, B: gaussian_rbf_kernel(A, B, sigma=sigma)

    ns = X.size(0)
    nt = Y.size(0)

    K_ss = kernel_fn(X, X)   # [ns, ns]
    K_tt = kernel_fn(Y, Y)   # [nt, nt]
    K_st = kernel_fn(X, Y)   # [ns, nt]

    return (
        K_ss.sum() / (ns * ns)
        - 2.0 * K_st.sum() / (ns * nt)
        + K_tt.sum() / (nt * nt)
    )


# -- MMDLoss --------------------------------------------------------------------

class MMDLoss(nn.Module):
    """
    Differentiable MMD loss for use inside training loops.

    Wraps ``compute_mmd_squared`` as an ``nn.Module`` so it can be
    composed with other losses and tracked by optimisers.

    Parameters
    ----------
    sigma : float | None
        Fixed kernel bandwidth.  When None (default) the bandwidth is
        recomputed each forward pass via the median heuristic — useful
        when the latent-space scale shifts during training.

    Example
    -------
    >>> mmd_loss = MMDLoss()
    >>> loss = mmd_loss(source_latent, target_latent)
    >>> loss.backward()
    """

    def __init__(self, sigma: Optional[float] = None) -> None:
        super().__init__()
        self.sigma = sigma

    def forward(
        self,
        source_latent: torch.Tensor,
        target_latent: torch.Tensor,
    ) -> torch.Tensor:
        """
        source_latent : [ns, d]  latent reps of generated (source-domain) samples
        target_latent : [nt, d]  latent reps of real target-domain samples
        -> scalar MMD² (differentiable w.r.t. source_latent and target_latent)
        """
        sigma = (
            self.sigma
            if self.sigma is not None
            else median_bandwidth(source_latent, target_latent)
        )
        kernel_fn: KernelFn = lambda A, B: gaussian_rbf_kernel(A, B, sigma=sigma)
        return compute_mmd_squared(source_latent, target_latent, kernel_fn)


# -- Pairwise dataset MMD matrix ------------------------------------------------

def compute_dataset_mmd_matrix(
    datasets: Dict[str, torch.Tensor],
    subsample: int = 1000,
    sigma: Optional[float] = None,
) -> pd.DataFrame:
    """
    Compute the symmetric pairwise MMD² matrix for a collection of datasets
    and save a heatmap to ``results/figures/mmd_distance_heatmap.png``.

    Parameters
    ----------
    datasets  : {name: [n, d] tensor}  –  one tensor per dataset (same d)
    subsample : int  –  randomly subsample each dataset to this size so the
                        O(n²) kernel computation stays tractable
    sigma     : float | None  –  shared fixed bandwidth; auto when None

    Returns
    -------
    pd.DataFrame  –  (n_datasets × n_datasets) symmetric MMD² matrix;
                     diagonal is 0 by definition
    """
    names = list(datasets.keys())
    n_ds = len(names)

    # Subsample for tractability
    subsampled: Dict[str, torch.Tensor] = {}
    for name, X in datasets.items():
        if X.size(0) > subsample:
            idx = torch.randperm(X.size(0), device=X.device)[:subsample]
            subsampled[name] = X[idx].float()
        else:
            subsampled[name] = X.float()

    # Fixed kernel_fn when sigma is provided; per-pair auto-sigma otherwise
    fixed_kernel: Optional[KernelFn] = (
        None
        if sigma is None
        else (lambda A, B: gaussian_rbf_kernel(A, B, sigma=sigma))
    )

    values = torch.zeros(n_ds, n_ds)
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            if i == j:
                values[i, j] = 0.0
            elif i > j:
                values[i, j] = values[j, i]  # exploit symmetry
            else:
                with torch.no_grad():
                    mmd2 = compute_mmd_squared(
                        subsampled[ni],
                        subsampled[nj],
                        kernel_fn=fixed_kernel,
                    )
                values[i, j] = mmd2.item()

    df = pd.DataFrame(values.numpy(), index=names, columns=names)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _save_heatmap(df, FIGURES_DIR / "mmd_distance_heatmap.png")

    return df


def _save_heatmap(df: pd.DataFrame, path: Path) -> None:
    """Save MMD² matrix as a colour-coded heatmap (seaborn preferred, matplotlib fallback)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(6, len(df) * 2), max(5, len(df) * 1.8)))

    try:
        import seaborn as sns
        sns.heatmap(
            df,
            annot=True,
            fmt=".4f",
            cmap="YlOrRd",
            ax=ax,
            linewidths=0.5,
            square=True,
            cbar_kws={"label": "MMD²"},
        )
    except ImportError:
        data = df.values.astype(float)
        im = ax.imshow(data, cmap="YlOrRd", aspect="auto")
        plt.colorbar(im, ax=ax, label="MMD²")
        ax.set_xticks(range(len(df.columns)))
        ax.set_yticks(range(len(df.index)))
        ax.set_xticklabels(df.columns, rotation=45, ha="right")
        ax.set_yticklabels(df.index)
        for r in range(len(df.index)):
            for c in range(len(df.columns)):
                ax.text(c, r, f"{data[r, c]:.4f}", ha="center", va="center", fontsize=9)

    ax.set_title("Pairwise MMD² Distance Between Datasets", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Heatmap saved -> {path}")


# -- Unit tests -----------------------------------------------------------------

if __name__ == "__main__":
    _GREEN = "\033[92m"
    _RED   = "\033[91m"
    _RESET = "\033[0m"

    def _label(ok: bool) -> str:
        return f"{_GREEN}PASS{_RESET}" if ok else f"{_RED}FAIL{_RESET}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEP = "-" * 56
    print(f"Device: {device}\n{SEP}")

    # -- Test 1: MMD between same distribution should be near zero -------------
    #
    # For X, Y ~ N(0, I_32) with n=1000 the biased estimator has
    #   E[MMD²_b] ≈ (2/n)(1 − E[k(x,x')]) ≈ 0.0008  (well below 0.01)
    # The empirical value should reliably fall under the threshold.
    print("Test 1: MMD²(N(0,I), N(0,I))  should be < 0.01")
    torch.manual_seed(42)
    X1 = torch.randn(1000, 32, device=device)
    Y1 = torch.randn(1000, 32, device=device)
    mmd1 = compute_mmd_squared(X1, Y1).item()
    ok1 = mmd1 < 0.01
    print(f"  MMD² = {mmd1:.6f}   threshold < 0.01   -> {_label(ok1)}\n")

    # -- Test 2: MMD between very different distributions should be large -------
    #
    # For X ~ N(0,I_32) and Y ~ N(10,I_32) the true MMD is large.
    # Regardless of bandwidth, the kernel between the two clouds is
    # negligible, giving MMD² ≈ k(X,X)/n² + k(Y,Y)/m² ≈ 2 * (1/n²)·n²·1 = 2.
    # A threshold of 0.5 is extremely conservative.
    print("Test 2: MMD²(N(0,I), N(10,I))  should be > 0.5")
    torch.manual_seed(7)
    X2 = torch.randn(500, 32, device=device)
    Y2 = torch.randn(500, 32, device=device) + 10.0
    mmd2 = compute_mmd_squared(X2, Y2).item()
    ok2 = mmd2 > 0.5
    print(f"  MMD² = {mmd2:.6f}   threshold > 0.50   -> {_label(ok2)}\n")

    # -- Test 3: Gradient flows through MMDLoss ---------------------------------
    #
    # MMDLoss.forward is a composition of differentiable ops (matmul, exp, sum);
    # sigma is extracted with .item() so it never enters the autograd graph.
    # After .backward() the gradient w.r.t. source_latent must be non-None
    # and finite.
    print("Test 3: MMDLoss should be differentiable (gradients must be non-None and finite)")
    src = torch.randn(128, 16, device=device, requires_grad=True)
    tgt = torch.randn(128, 16, device=device)

    loss_fn = MMDLoss()
    loss = loss_fn(src, tgt)
    loss.backward()

    has_grad = src.grad is not None
    is_finite = has_grad and torch.isfinite(src.grad).all().item()
    ok3 = has_grad and is_finite
    grad_norm = src.grad.norm().item() if has_grad else float("nan")
    print(f"  loss = {loss.item():.6f}   grad_norm = {grad_norm:.6f}   -> {_label(ok3)}\n")

    # Summary
    all_pass = ok1 and ok2 and ok3
    results = [("Test 1 (same dist)",   ok1),
               ("Test 2 (diff dist)",   ok2),
               ("Test 3 (gradient)",    ok3)]
    print(SEP)
    for name, ok in results:
        print(f"  {name:<24} {_label(ok)}")
    print(SEP)
    print(f"  Overall: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")

    # -- Optional: pairwise matrix demo ----------------------------------------
    print("\nDemo: compute_dataset_mmd_matrix with synthetic datasets")
    torch.manual_seed(0)
    demo_datasets = {
        "NSL-KDD (sim)":    torch.randn(300, 16, device=device),
        "UNSW-NB15 (sim)":  torch.randn(300, 16, device=device) + 2.0,
        "CIC-IDS (sim)":    torch.randn(300, 16, device=device) + 5.0,
    }
    df = compute_dataset_mmd_matrix(demo_datasets, subsample=300)
    print(df.to_string(float_format=lambda x: f"{x:.4f}"))
