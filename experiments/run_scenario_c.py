"""
Scenario C: Adapted transfer with MMD latent-space alignment.

Trains CrossDatasetCEGAN (CE-GAN + MMD loss) on the source dataset while
providing aligned target samples for domain regularisation. Maps generated
samples to the target space via FeatureHarmonizer, augments the target
training set, and evaluates a Nash ensemble classifier.

Also computes the Transferability Quality Score (TQS) by comparing against
a no-augmentation baseline on both source and target domains.

Usage
-----
  python experiments/run_scenario_c.py --source nsl_kdd --target unsw_nb15
  python experiments/run_scenario_c.py --source nsl_kdd --target cic_ids2017 \\
      --epochs 100 --gamma 0.05
  python experiments/run_scenario_c.py --source nsl_kdd --target unsw_nb15 \\
      --synthetic --epochs 5   # quick smoke-test
"""

import argparse
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments._exp_utils import (
    DS_CONFIG, get_data, train_gan, augment_training, print_metrics_summary,
)
from src.cegan import CrossDatasetCEGAN, CEGANConfig
from src.alignment import FeatureHarmonizer
from src.classifier import NashEnsembleClassifier, compute_all_metrics, save_results_table
from src.metrics import compute_tqs, save_tqs_table
from sklearn.metrics import f1_score


# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Scenario C: adapted transfer with MMD alignment"
    )
    p.add_argument("--source",     default="nsl_kdd",   choices=list(DS_CONFIG))
    p.add_argument("--target",     default="unsw_nb15", choices=list(DS_CONFIG))
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--batch_size", type=int,   default=256)
    p.add_argument("--gamma",      type=float, default=0.05, help="MMD loss weight")
    p.add_argument("--n_aug",      type=int,   default=200,  help="Synthetic samples per class")
    p.add_argument("--d_model",    type=int,   default=128)
    p.add_argument("--n_layers",   type=int,   default=3)
    p.add_argument("--n_estimators", type=int, default=100)
    p.add_argument("--projector_epochs", type=int, default=200)
    p.add_argument("--synthetic",  action="store_true", help="Force synthetic data")
    p.add_argument("--save_ckpt",  action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _align_target_to_source(
    X_tgt: torch.Tensor, n_feat_src: int
) -> torch.Tensor:
    """
    Bring target tensor to source feature dimensionality for latent-space MMD.
    Uses zero-padding (if target has fewer features) or truncation (if more).
    """
    n_tgt = X_tgt.size(1)
    if n_tgt == n_feat_src:
        return X_tgt
    if n_tgt < n_feat_src:
        pad = torch.zeros(X_tgt.size(0), n_feat_src - n_tgt, dtype=X_tgt.dtype)
        return torch.cat([X_tgt, pad], dim=1)
    return X_tgt[:, :n_feat_src]


def _macro_f1(y_true, y_pred) -> float:
    """Compute macro-averaged F1 from arrays or tensors."""
    from src.classifier import _labels_to_numpy
    return float(f1_score(
        _labels_to_numpy(y_true),
        _labels_to_numpy(y_pred),
        average="macro",
        zero_division=0,
    ))


# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    if args.source == args.target:
        print("Error: --source and --target must be different datasets")
        sys.exit(1)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*60}")
    print(f"  Scenario C: adapted transfer (MMD)  |  {args.source} -> {args.target}")
    print(f"  Device: {device_str}  |  epochs={args.epochs}  |  gamma={args.gamma}  |  n_aug={args.n_aug}")
    print(f"{'='*60}\n")

    # -- Load data -------------------------------------------------------------
    print("Loading source data...")
    X_src_train, X_src_test, y_src_train, y_src_test, n_feat_src, n_cls_src, src_df, _ = get_data(
        args.source, force_synthetic=args.synthetic
    )
    print("Loading target data...")
    X_tgt_train, X_tgt_test, y_tgt_train, y_tgt_test, n_feat_tgt, n_cls_tgt, tgt_df, tgt_cls = get_data(
        args.target, force_synthetic=args.synthetic
    )

    # -- Fit FeatureHarmonizer -------------------------------------------------
    print(f"\nFitting FeatureHarmonizer ({args.source} -> {args.target})...")
    harmonizer = FeatureHarmonizer(args.source, args.target)
    harmonizer.fit(src_df, tgt_df, projector_epochs=args.projector_epochs)
    s = harmonizer.summary()
    print(f"  Semantic: {s['n_semantic']}  |  Projected: {s['n_projected_map']}  "
          f"|  FC-filled: {s['n_fc_projected']}  |  "
          f"Covered: {s['n_total_covered']}/{s['n_target_feat']}")

    # -- Align target to source feature space (for MMD during training) --------
    X_tgt_aligned = _align_target_to_source(X_tgt_train, n_feat_src)
    print(f"\nTarget samples aligned to source space: "
          f"{tuple(X_tgt_train.shape)} -> {tuple(X_tgt_aligned.shape)}")

    # -- Train CrossDatasetCEGAN -----------------------------------------------
    print(f"\nTraining CrossDatasetCEGAN on {args.source} (gamma={args.gamma})...")
    cfg = CEGANConfig(
        n_features=n_feat_src, n_classes=n_cls_src,
        d_model=args.d_model, n_layers=args.n_layers,
        device=device_str,
    )
    model = CrossDatasetCEGAN(cfg, gamma=args.gamma)
    train_gan(
        model, X_src_train, y_src_train,
        epochs=args.epochs, batch_size=args.batch_size,
        mode="cross", X_tgt=X_tgt_aligned,
        label="CrossGAN",
    )

    if args.save_ckpt:
        ckpt = _PROJECT_ROOT / "results" / "checkpoints" / f"scen_c_{args.source}_{args.target}_final.pt"
        model.save_checkpoint(ckpt)
        print(f"  Checkpoint -> {ckpt.name}")

    # -- Generate source-domain synthetic samples ------------------------------
    print(f"\nGenerating {args.n_aug} synthetic samples per class on source domain...")
    X_src_aug, y_src_aug = augment_training(model, X_src_train, y_src_train, args.n_aug)
    n_real_src = X_src_train.shape[0]
    X_synth_src = X_src_aug[n_real_src:].to(model.device)
    y_synth     = y_src_aug[n_real_src:]

    # -- Map synthetic samples to target feature space -------------------------
    print(f"Mapping synthetic samples to {args.target} feature space...")
    X_synth_tgt = harmonizer.transform(X_synth_src).cpu()
    print(f"  Source: {tuple(X_synth_src.shape)}  ->  Target: {tuple(X_synth_tgt.shape)}")

    y_synth_tgt = y_synth.clamp(0, n_cls_tgt - 1)

    # -- Augment target training set -------------------------------------------
    X_tgt_aug = torch.cat([X_tgt_train, X_synth_tgt], dim=0)
    y_tgt_aug = torch.cat([y_tgt_train, y_synth_tgt], dim=0)
    print(f"\nTarget training set: {X_tgt_train.shape[0]:,} real + "
          f"{X_synth_tgt.shape[0]:,} synthetic = {X_tgt_aug.shape[0]:,} total")

    # -- Nash classifier on augmented target data (cross-dataset, equal weights)
    print("\nTraining Nash classifier on augmented target data...")
    clf_tgt = NashEnsembleClassifier(
        init_equal_weights=True,
        n_estimators=args.n_estimators,
        verbose=True,
    )
    clf_tgt.fit(X_tgt_aug, y_tgt_aug)
    y_pred_tgt = clf_tgt.predict(X_tgt_test)
    metrics_c = compute_all_metrics(y_tgt_test, y_pred_tgt, class_names=tgt_cls)
    f1_target_aug = metrics_c["overall"]["macro_f1"]

    # -- Save Scenario C results -----------------------------------------------
    csv_path = save_results_table(
        metrics_c, scenario="C", source=args.source, target=args.target
    )
    print(f"\nResults appended -> {csv_path}")

    # =========================================================================
    # TQS computation
    # Needs four F1 values:
    #   f1_target_aug   : already computed above
    #   f1_target_base  : target test F1 WITHOUT augmentation
    #   f1_source_within: source test F1 WITH CE-GAN augmentation (reuse current model)
    #   f1_source_base  : source test F1 WITHOUT augmentation
    # =========================================================================
    print("\n--- Computing TQS components ---")

    # f1_target_base: train Nash on original target data (no augmentation)
    print("  [TQS] target baseline (no augmentation)...")
    clf_tgt_base = NashEnsembleClassifier(
        init_equal_weights=True, n_estimators=args.n_estimators, verbose=False
    )
    clf_tgt_base.fit(X_tgt_train, y_tgt_train)
    f1_target_base = _macro_f1(y_tgt_test, clf_tgt_base.predict(X_tgt_test))

    # f1_source_within: augment source with the trained CrossDatasetCEGAN, then classify
    # (The model was trained on source with MMD regularisation; its generator
    #  is still conditioned on source labels and produces source-space samples.)
    print("  [TQS] source within-dataset (augmented with trained model)...")
    clf_src_within = NashEnsembleClassifier(
        init_equal_weights=False, n_estimators=args.n_estimators, verbose=False
    )
    clf_src_within.fit(X_src_aug, y_src_aug)   # already computed above
    f1_source_within = _macro_f1(y_src_test, clf_src_within.predict(X_src_test))

    # f1_source_base: train Nash on original source data (no augmentation)
    print("  [TQS] source baseline (no augmentation)...")
    clf_src_base = NashEnsembleClassifier(
        init_equal_weights=False, n_estimators=args.n_estimators, verbose=False
    )
    clf_src_base.fit(X_src_train, y_src_train)
    f1_source_base = _macro_f1(y_src_test, clf_src_base.predict(X_src_test))

    print(f"\n  F1 target augmented : {f1_target_aug:.4f}")
    print(f"  F1 target baseline  : {f1_target_base:.4f}")
    print(f"  F1 source within    : {f1_source_within:.4f}")
    print(f"  F1 source baseline  : {f1_source_base:.4f}")

    tqs, interpretation = compute_tqs(
        f1_target_aug, f1_target_base, f1_source_within, f1_source_base
    )

    tqs_str = f"{tqs:.2f}" if tqs is not None else "N/A"
    print(f"\n  TQS = {tqs_str}   ->  {interpretation}")

    # Save TQS table
    tqs_path = save_tqs_table(
        {
            (args.source, args.target): {
                "scenario_b_tqs":  None,   # not computed in this script
                "scenario_c_tqs":  tqs,
                "interpretation":  interpretation,
            }
        }
    )

    # -- Final summary ---------------------------------------------------------
    print_metrics_summary(
        f"Scenario C  |  {args.source} -> {args.target}  |  MMD-aligned transfer",
        metrics_c,
    )
    print(f"\nScenario C complete.")
    print(f"  Macro F1 (target, augmented)  : {f1_target_aug:.4f}")
    print(f"  Macro F1 (target, baseline)   : {f1_target_base:.4f}")
    print(f"  TQS                           : {tqs_str}  ({interpretation})")


if __name__ == "__main__":
    main()
