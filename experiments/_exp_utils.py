"""
Shared utilities for scenario A/B/C experiment scripts.

Not a public module -- imported only by the three run_scenario_*.py scripts.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data_utils

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Dataset dimension registry (used only for synthetic fallback)
DS_CONFIG: dict = {
    "nsl_kdd":     {"n_features": 41,  "n_classes": 5,  "max_samples": None},
    "unsw_nb15":   {"n_features": 43,  "n_classes": 10, "max_samples": 100_000},
    "cic_ids2017": {"n_features": 78,  "n_classes": 15, "max_samples": 100_000},
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_data(name: str, force_synthetic: bool = False):
    """
    Load a dataset (real or synthetic fallback).

    Returns
    -------
    X_train, X_test, y_train, y_test : CPU tensors
    n_features, n_classes            : ints
    df_for_harmonizer                : pd.DataFrame with feature column names
    class_names                      : list[str] | None
    """
    cfg        = DS_CONFIG[name]
    n_features = cfg["n_features"]
    n_classes  = cfg["n_classes"]

    if not force_synthetic:
        try:
            from src.datasets import DatasetLoader
            max_s = cfg.get("max_samples")
            loader = DatasetLoader(name, max_samples=max_s)
            X_train, X_test, y_train, y_test = loader.load()
            X_train, X_test = X_train.cpu(), X_test.cpu()
            y_train, y_test = y_train.cpu(), y_test.cpu()
            n_features = loader.num_features
            n_classes  = loader.num_classes
            class_names = loader.class_names
            df = _make_df(X_train, name, n_features)
            total = X_train.shape[0] + X_test.shape[0]
            print(f"  Loaded  {name}: {total:,} samples  x{n_features} feat  {n_classes} cls")
            return X_train, X_test, y_train, y_test, n_features, n_classes, df, class_names
        except Exception as exc:
            print(f"  [{name}] Falling back to synthetic data ({type(exc).__name__}: {exc})")

    # Synthetic fallback: stratified-like balanced split
    n_per_class = max(400, 4000 // n_classes)
    Xs, ys = [], []
    rng = np.random.default_rng(42)
    for c in range(n_classes):
        Xs.append(rng.random((n_per_class, n_features)).astype(np.float32) + c * 0.1)
        ys.append(np.full(n_per_class, c, dtype=np.int64))
    X_all = np.concatenate(Xs)
    y_all = np.concatenate(ys)
    perm = rng.permutation(len(y_all))
    X_all, y_all = X_all[perm], y_all[perm]
    split = int(0.8 * len(y_all))
    X_train = torch.tensor(X_all[:split])
    X_test  = torch.tensor(X_all[split:])
    y_train = torch.tensor(y_all[:split])
    y_test  = torch.tensor(y_all[split:])
    df = _make_df(X_train, name, n_features)
    print(f"  Synthetic {name}: {len(y_all)} samples  x{n_features} feat  {n_classes} cls")
    return X_train, X_test, y_train, y_test, n_features, n_classes, df, None


def _make_df(X: torch.Tensor, name: str, n_features: int) -> pd.DataFrame:
    """Build a DataFrame with proper column names for FeatureHarmonizer.fit()."""
    from src.alignment import DATASET_FEATURES
    cols = DATASET_FEATURES[name][:n_features]
    return pd.DataFrame(X.numpy()[:, :len(cols)], columns=cols)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_loader(
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    shuffle: bool = True,
) -> data_utils.DataLoader:
    ds = data_utils.TensorDataset(X, y)
    return data_utils.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, drop_last=True
    )


# ---------------------------------------------------------------------------
# GAN training loop
# ---------------------------------------------------------------------------

def train_gan(
    model,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int,
    batch_size: int,
    mode: str = "baseline",      # "baseline" | "cross"
    X_tgt: Optional[torch.Tensor] = None,
    log_every: int = 25,
    label: str = "",
) -> None:
    """
    Train a CEGAN (mode='baseline') or CrossDatasetCEGAN (mode='cross') in-place.

    For cross mode, X_tgt must be pre-aligned to the source feature space
    (same n_features as X_train).
    """
    src_loader = make_loader(X_train, y_train, batch_size)

    tgt_iter = None
    if mode == "cross":
        if X_tgt is None:
            raise ValueError("train_gan: X_tgt is required for mode='cross'")
        tgt_loader = make_loader(
            X_tgt,
            torch.zeros(X_tgt.size(0), dtype=torch.long),
            batch_size,
        )
        tgt_iter = iter(tgt_loader)

    prefix = f"  [{label}] " if label else "  "
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in src_loader:
            if mode == "baseline":
                model.train_step(xb, yb)
            else:
                try:
                    xt, _ = next(tgt_iter)
                except StopIteration:
                    tgt_iter = iter(tgt_loader)
                    xt, _ = next(tgt_iter)
                model.train_cross_dataset_step(xb, yb, xt)

        if epoch % log_every == 0 or epoch == epochs:
            print(f"{prefix}epoch {epoch:4d}/{epochs}")


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def augment_training(
    model,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    n_aug_per_class: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate n_aug_per_class synthetic samples per class and append to training set.

    Returns CPU tensors (X_aug, y_aug).
    """
    classes = y_train.unique().tolist()
    extra_X, extra_y = [], []
    for cls in classes:
        synth = model.generate(n_samples=n_aug_per_class, class_label=int(cls))
        extra_X.append(synth.cpu())
        extra_y.append(torch.full((n_aug_per_class,), int(cls), dtype=torch.long))
    return (
        torch.cat([X_train] + extra_X, dim=0),
        torch.cat([y_train] + extra_y, dim=0),
    )


# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------

def print_metrics_summary(title: str, metrics: dict) -> None:
    ov  = metrics["overall"]
    sep = "=" * 58
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    print(f"  Accuracy    : {ov['accuracy']:.4f}")
    print(f"  Macro F1    : {ov['macro_f1']:.4f}")
    print(f"  Weighted F1 : {ov['weighted_f1']:.4f}")
    print(f"  MCC         : {ov['mcc']:.4f}")
    if metrics.get("per_class"):
        print(f"\n  Per-class F1  (name / f1 / support):")
        print(f"  {'Class':<12} {'F1':>7} {'Supp':>6}")
        print("  " + "-" * 28)
        for e in metrics["per_class"]:
            name = e.get("name", f"c{e['class_id']}")
            print(f"  {name:<12} {e['f1']:7.4f} {e['support']:6d}")
    print(sep)
