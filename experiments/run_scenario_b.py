"""
Scenario B: Direct transfer without MMD alignment.

Trains CE-GAN on the source dataset, maps generated samples to the target
feature space via FeatureHarmonizer (semantic mapping + FC projection),
injects them into the target training set, and evaluates a Nash ensemble
classifier on the target test set.

Usage
-----
  python experiments/run_scenario_b.py --source nsl_kdd --target unsw_nb15
  python experiments/run_scenario_b.py --source nsl_kdd --target cic_ids2017 --epochs 100
  python experiments/run_scenario_b.py --source nsl_kdd --target unsw_nb15 --synthetic
"""

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments._exp_utils import (
    DS_CONFIG, get_data, train_gan, augment_training, print_metrics_summary,
)
from src.cegan import CEGAN, CEGANConfig
from src.alignment import FeatureHarmonizer
from src.classifier import NashEnsembleClassifier, compute_all_metrics, save_results_table


# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scenario B: direct transfer (no MMD alignment)")
    p.add_argument("--source",     default="nsl_kdd",  choices=list(DS_CONFIG))
    p.add_argument("--target",     default="unsw_nb15", choices=list(DS_CONFIG))
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--batch_size", type=int,   default=256)
    p.add_argument("--n_aug",      type=int,   default=200, help="Synthetic samples per class")
    p.add_argument("--d_model",    type=int,   default=128)
    p.add_argument("--n_layers",   type=int,   default=3)
    p.add_argument("--n_estimators", type=int, default=100)
    p.add_argument("--projector_epochs", type=int, default=200,
                   help="FeatureHarmonizer projector training epochs")
    p.add_argument("--synthetic",  action="store_true", help="Force synthetic data")
    p.add_argument("--save_ckpt",  action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    if args.source == args.target:
        print("Error: --source and --target must be different datasets")
        sys.exit(1)

    device_str = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    print(f"\n{'='*58}")
    print(f"  Scenario B: direct transfer  |  {args.source} -> {args.target}")
    print(f"  Device: {device_str}  |  epochs={args.epochs}  |  n_aug={args.n_aug}")
    print(f"{'='*58}\n")

    # -- Load data -------------------------------------------------------------
    print("Loading source data...")
    X_src_train, _, y_src_train, _, n_feat_src, n_cls_src, src_df, _ = get_data(
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

    # -- Train CE-GAN on source ------------------------------------------------
    print(f"\nTraining CE-GAN on {args.source}...")
    cfg = CEGANConfig(
        n_features=n_feat_src, n_classes=n_cls_src,
        d_model=args.d_model, n_layers=args.n_layers,
        device=device_str,
    )
    model = CEGAN(cfg)
    train_gan(model, X_src_train, y_src_train, args.epochs, args.batch_size, label="GAN-src")

    if args.save_ckpt:
        ckpt = _PROJECT_ROOT / "results" / "checkpoints" / f"scen_b_{args.source}_final.pt"
        model.save_checkpoint(ckpt)
        print(f"  Checkpoint -> {ckpt.name}")

    # -- Generate source-domain synthetic samples ------------------------------
    print(f"\nGenerating {args.n_aug} synthetic samples per class on source domain...")
    X_src_aug, y_src_aug = augment_training(model, X_src_train, y_src_train, args.n_aug)
    # Keep only the NEW synthetic samples (exclude real source)
    n_real = X_src_train.shape[0]
    X_synth_src = X_src_aug[n_real:].to(model.device)   # [n_cls * n_aug, n_feat_src]
    y_synth     = y_src_aug[n_real:]                     # CPU

    # -- Map synthetic samples to target feature space -------------------------
    print(f"Mapping synthetic samples to {args.target} feature space via harmonizer...")
    X_synth_tgt = harmonizer.transform(X_synth_src).cpu()
    print(f"  Source shape: {tuple(X_synth_src.shape)}  ->  "
          f"Target shape: {tuple(X_synth_tgt.shape)}")

    # Remap labels: source labels -> target label space
    # (labels are integers 0..n_cls-1; if source has fewer classes, clamp)
    import torch
    y_synth_tgt = y_synth.clamp(0, n_cls_tgt - 1)

    # -- Augment target training set -------------------------------------------
    X_tgt_aug = torch.cat([X_tgt_train, X_synth_tgt], dim=0)
    y_tgt_aug = torch.cat([y_tgt_train, y_synth_tgt], dim=0)
    print(f"\nTarget training set: {X_tgt_train.shape[0]:,} real + "
          f"{X_synth_tgt.shape[0]:,} synthetic = {X_tgt_aug.shape[0]:,} total")

    # -- Nash classifier (equal weights for cross-dataset) ---------------------
    print("\nTraining Nash ensemble classifier on augmented target data...")
    clf = NashEnsembleClassifier(
        init_equal_weights=True,    # cross-dataset: no prior domain bias
        n_estimators=args.n_estimators,
        verbose=True,
    )
    clf.fit(X_tgt_aug, y_tgt_aug)
    y_pred = clf.predict(X_tgt_test)
    metrics = compute_all_metrics(y_tgt_test, y_pred, class_names=tgt_cls)

    # -- Save results ----------------------------------------------------------
    csv_path = save_results_table(
        metrics, scenario="B", source=args.source, target=args.target
    )
    print(f"\nResults appended -> {csv_path}")

    # -- Summary ---------------------------------------------------------------
    print_metrics_summary(
        f"Scenario B  |  {args.source} -> {args.target}  |  direct transfer", metrics
    )
    print(f"\nScenario B complete. Macro F1: {metrics['overall']['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
