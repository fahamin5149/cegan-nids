"""
Scenario A: Within-dataset baseline.

Trains CE-GAN on a single dataset, augments the training set with synthetic
minority-class samples, trains a Nash ensemble classifier on the augmented set,
and evaluates on the held-out test set.

Usage
-----
  python experiments/run_scenario_a.py --dataset nsl_kdd
  python experiments/run_scenario_a.py --dataset unsw_nb15 --epochs 100 --n_aug 300
  python experiments/run_scenario_a.py --dataset nsl_kdd --synthetic   # smoke-test
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
from src.classifier import NashEnsembleClassifier, compute_all_metrics, save_results_table


# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scenario A: within-dataset CE-GAN baseline")
    p.add_argument("--dataset",    default="nsl_kdd", choices=list(DS_CONFIG))
    p.add_argument("--epochs",     type=int,   default=100,  help="CE-GAN training epochs")
    p.add_argument("--batch_size", type=int,   default=256)
    p.add_argument("--n_aug",      type=int,   default=200,  help="Synthetic samples per class")
    p.add_argument("--d_model",    type=int,   default=128)
    p.add_argument("--n_layers",   type=int,   default=3)
    p.add_argument("--n_estimators", type=int, default=100,  help="Trees per classifier")
    p.add_argument("--synthetic",  action="store_true",      help="Force synthetic data")
    p.add_argument("--save_ckpt",  action="store_true",      help="Save GAN checkpoint")
    return p.parse_args()


# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device_str = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    print(f"\n{'='*58}")
    print(f"  Scenario A: within-dataset  |  {args.dataset}")
    print(f"  Device: {device_str}  |  epochs={args.epochs}  |  n_aug={args.n_aug}")
    print(f"{'='*58}\n")

    # -- Data ------------------------------------------------------------------
    print("Loading data...")
    X_train, X_test, y_train, y_test, n_feat, n_cls, _, class_names = get_data(
        args.dataset, force_synthetic=args.synthetic
    )

    # -- Train CE-GAN ----------------------------------------------------------
    print("\nTraining CE-GAN...")
    cfg = CEGANConfig(
        n_features=n_feat, n_classes=n_cls,
        d_model=args.d_model, n_layers=args.n_layers,
        device=device_str,
    )
    model = CEGAN(cfg)
    train_gan(model, X_train, y_train, args.epochs, args.batch_size, label="GAN")

    if args.save_ckpt:
        ckpt = _PROJECT_ROOT / "results" / "checkpoints" / f"scen_a_{args.dataset}_final.pt"
        model.save_checkpoint(ckpt)
        print(f"  Checkpoint -> {ckpt.name}")

    # -- Augment training set --------------------------------------------------
    print(f"\nGenerating {args.n_aug} synthetic samples per class ({n_cls} classes)...")
    X_aug, y_aug = augment_training(model, X_train, y_train, args.n_aug)
    print(f"  Training set: {X_train.shape[0]:,} real + {X_aug.shape[0]-X_train.shape[0]:,} "
          f"synthetic = {X_aug.shape[0]:,} total")

    # -- Nash classifier on augmented data -------------------------------------
    print("\nTraining Nash ensemble classifier (augmented data)...")
    clf_aug = NashEnsembleClassifier(
        init_equal_weights=False,   # within-dataset: performance-based init
        n_estimators=args.n_estimators,
        verbose=True,
    )
    clf_aug.fit(X_aug, y_aug)
    y_pred = clf_aug.predict(X_test)
    metrics = compute_all_metrics(y_test, y_pred, class_names=class_names)

    # -- Save results ----------------------------------------------------------
    csv_path = save_results_table(
        metrics, scenario="A", source=args.dataset, target=args.dataset
    )
    print(f"\nResults appended -> {csv_path}")

    # -- Summary ---------------------------------------------------------------
    print_metrics_summary(
        f"Scenario A  |  {args.dataset}  |  augmented with CE-GAN", metrics
    )
    print(f"\nScenario A complete. Macro F1: {metrics['overall']['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
