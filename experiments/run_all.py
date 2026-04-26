"""
run_all.py -- Full experimental pipeline orchestrator.

Runs all scenarios in order, computes TQS for every source→target pair,
benchmarks SMOTE, and prints a side-by-side summary table.

Usage
-----
  # Full run with real data (long)
  python experiments/run_all.py

  # Quick smoke-test with synthetic data
  python experiments/run_all.py --dry_run

  # Synthetic data, custom epochs
  python experiments/run_all.py --synthetic --epochs 10 --n_aug 50

  # Skip SMOTE (e.g. imbalanced-learn not installed)
  python experiments/run_all.py --dry_run --skip_smote
"""

import argparse
import csv
import logging
import os
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE         = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

RESULTS_DIR = _PROJECT_ROOT / "results"
TABLES_DIR  = RESULTS_DIR / "tables"
LOG_PATH    = RESULTS_DIR / "experiment_log.txt"
PYTHON      = sys.executable   # same interpreter / venv as the orchestrator

# ---------------------------------------------------------------------------
# Experiment scope
# ---------------------------------------------------------------------------
DATASETS = ["nsl_kdd", "unsw_nb15", "cic_ids2017"]

PAIRS: List[Tuple[str, str]] = [
    ("nsl_kdd",   "unsw_nb15"),
    ("unsw_nb15", "nsl_kdd"),
    ("nsl_kdd",   "cic_ids2017"),
    ("unsw_nb15", "cic_ids2017"),
]

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CE-GAN full experimental pipeline"
    )
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--batch_size", type=int,   default=256)
    p.add_argument("--n_aug",      type=int,   default=200,
                   help="Synthetic samples per class per scenario")
    p.add_argument("--d_model",    type=int,   default=128)
    p.add_argument("--n_layers",   type=int,   default=3)
    p.add_argument("--n_estimators", type=int, default=100)
    p.add_argument("--projector_epochs", type=int, default=200,
                   help="FeatureHarmonizer FC-projector training epochs")
    p.add_argument("--synthetic",  action="store_true",
                   help="Force synthetic data (skip real CSV loading)")
    p.add_argument("--skip_smote", action="store_true",
                   help="Skip SMOTE baseline (use if imblearn not installed)")
    p.add_argument("--dry_run",    action="store_true",
                   help="Shorthand for --synthetic --epochs 3 --n_aug 20 "
                        "--projector_epochs 20")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging() -> logging.Logger:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("run_all")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s",
                             datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------

def _run_script(
    script: str,
    extra_args: List[str],
    step_name: str,
    logger: logging.Logger,
) -> Tuple[bool, str]:
    """
    Run a script via subprocess.  Stream output to console + log file in
    real-time.  Returns (success, full_stdout_text).
    """
    cmd = [PYTHON, str(_HERE / script)] + extra_args
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    sep = "-" * 60
    logger.info("%s", sep)
    logger.info("STEP: %s", step_name)
    logger.info("CMD : %s", " ".join(cmd))
    logger.info("%s", sep)

    lines: List[str] = []
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        for raw_line in proc.stdout:
            line = raw_line.rstrip()
            lines.append(line)
            # Write to log file (via logger at DEBUG so it goes to file only)
            logging.getLogger("run_all.sub").debug("%s", line)
            # Echo to console without the timestamp prefix
            print(line)
        proc.wait()
        stdout = "\n".join(lines)
        if proc.returncode == 0:
            logger.info("DONE: %s (exit 0)\n", step_name)
            return True, stdout
        else:
            logger.error("FAILED: %s (exit %d)\n", step_name, proc.returncode)
            return False, stdout
    except Exception as exc:
        logger.error("EXCEPTION in %s: %s\n", step_name, exc)
        return False, ""


def _parse_f1(stdout: str, pattern: str) -> Optional[float]:
    m = re.search(pattern, stdout)
    return float(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Shared CLI args string builder
# ---------------------------------------------------------------------------

def _common_flags(args: argparse.Namespace) -> List[str]:
    flags = [
        "--epochs",     str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--n_aug",      str(args.n_aug),
        "--d_model",    str(args.d_model),
        "--n_layers",   str(args.n_layers),
        "--n_estimators", str(args.n_estimators),
    ]
    if args.synthetic:
        flags.append("--synthetic")
    return flags


# ---------------------------------------------------------------------------
# Inline: no-augmentation baseline (Nash only, no GAN)
# ---------------------------------------------------------------------------

def _no_aug_f1(
    dataset: str,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> Optional[float]:
    """
    Train Nash classifier on raw training data (no GAN augmentation).
    Returns macro-F1 on the test split.
    """
    step = f"no-aug baseline [{dataset}]"
    logger.info("Running %s ...", step)
    try:
        from experiments._exp_utils import get_data
        from src.classifier import NashEnsembleClassifier
        from sklearn.metrics import f1_score

        X_tr, X_te, y_tr, y_te, _, _, _, _ = get_data(
            dataset, force_synthetic=args.synthetic
        )
        clf = NashEnsembleClassifier(
            init_equal_weights=False,
            n_estimators=args.n_estimators,
            verbose=False,
        )
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        from src.classifier import _labels_to_numpy
        score = float(f1_score(
            _labels_to_numpy(y_te), y_pred, average="macro", zero_division=0
        ))
        logger.info("  %s  macro-F1 = %.4f", step, score)
        return score
    except Exception as exc:
        logger.error("  %s FAILED: %s", step, exc)
        return None


# ---------------------------------------------------------------------------
# Inline: SMOTE baseline
# ---------------------------------------------------------------------------

def _run_smote(
    source: str,
    target: str,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> Optional[Dict]:
    """
    Apply SMOTE to the TARGET training set, train Nash, evaluate.
    Returns the compute_all_metrics dict or None on failure.
    """
    step = f"SMOTE baseline [{source} -> {target}]"
    logger.info("Running %s ...", step)

    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        logger.warning("  imbalanced-learn not installed — skipping SMOTE")
        return None

    try:
        from collections import Counter
        import numpy as np
        from experiments._exp_utils import get_data
        from src.classifier import NashEnsembleClassifier, compute_all_metrics

        _, _, _, _, _, _, _, _ = get_data(source, force_synthetic=args.synthetic)   # warm-up only
        X_tr, X_te, y_tr, y_te, _, n_cls, _, tgt_cls = get_data(
            target, force_synthetic=args.synthetic
        )

        X_np = X_tr.numpy()
        y_np = y_tr.numpy()

        counts  = Counter(y_np.tolist())
        min_cnt = min(counts.values())
        k = min(5, min_cnt - 1)
        if k < 1:
            logger.warning("  %s: min class count=%d < 2 — skipping", step, min_cnt)
            return None

        smote = SMOTE(random_state=42, k_neighbors=k)
        X_res, y_res = smote.fit_resample(X_np, y_np)
        logger.info("  SMOTE: %d -> %d samples", len(y_np), len(y_res))

        clf = NashEnsembleClassifier(
            init_equal_weights=True,
            n_estimators=args.n_estimators,
            verbose=False,
        )
        clf.fit(X_res, y_res)
        metrics = compute_all_metrics(y_te, clf.predict(X_te), class_names=tgt_cls)
        ov = metrics["overall"]
        logger.info(
            "  %s  acc=%.4f  macro_f1=%.4f", step, ov["accuracy"], ov["macro_f1"]
        )
        return metrics
    except Exception as exc:
        logger.error("  %s FAILED: %s", step, exc)
        return None


# ---------------------------------------------------------------------------
# Save SMOTE comparison table
# ---------------------------------------------------------------------------

def _save_smote_table(smote_results: Dict, logger: logging.Logger) -> Path:
    path = TABLES_DIR / "table5_smote_comparison.csv"
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    fields = [
        "source_dataset", "target_dataset",
        "smote_accuracy", "smote_macro_f1", "smote_weighted_f1",
        "smote_minority_class_f1",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (src, tgt), metrics in smote_results.items():
            if metrics is None:
                w.writerow({
                    "source_dataset": src, "target_dataset": tgt,
                    "smote_accuracy": "N/A", "smote_macro_f1": "N/A",
                    "smote_weighted_f1": "N/A", "smote_minority_class_f1": "N/A",
                })
                continue
            ov = metrics["overall"]
            pc = metrics.get("per_class", [])
            minority = min(pc, key=lambda e: e["support"]) if pc else {}
            w.writerow({
                "source_dataset":        src,
                "target_dataset":        tgt,
                "smote_accuracy":        f"{ov['accuracy']:.6f}",
                "smote_macro_f1":        f"{ov['macro_f1']:.6f}",
                "smote_weighted_f1":     f"{ov['weighted_f1']:.6f}",
                "smote_minority_class_f1": f"{minority.get('f1', float('nan')):.6f}",
            })
    logger.info("SMOTE table saved -> %s", path)
    return path


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def _print_summary(
    scen_a:   Dict[str, Optional[float]],
    scen_b:   Dict[Tuple, Optional[float]],
    scen_c:   Dict[Tuple, Optional[float]],
    smote:    Dict[Tuple, Optional[Dict]],
    base_f1:  Dict[str, Optional[float]],
    tqs_dict: Dict,
    failed:   List[str],
) -> None:
    def _f(v, width=7):
        return f"{v:.4f}".rjust(width) if v is not None else "  N/A ".rjust(width)

    def _t(v, width=7):
        return f"{v:.1f}".rjust(width) if v is not None else "  N/A ".rjust(width)

    sep_full  = "=" * 98
    sep_mid   = "-" * 98
    hdr_pair  = f"{'Pair':<28}"
    hdr_cols  = (
        f"{'A-src':>7}  {'B-tgt':>7}  {'C-tgt':>7}  "
        f"{'SMOTE':>7}  {'base-src':>8}  {'base-tgt':>8}  "
        f"{'TQS-B':>7}  {'TQS-C':>7}  {'Interpretation'}"
    )
    print(f"\n{sep_full}")
    print("  FINAL RESULTS SUMMARY")
    print(sep_full)
    print(f"  {hdr_pair} {hdr_cols}")
    print(sep_mid)

    for src, tgt in PAIRS:
        pair_label = f"{src} -> {tgt}"
        smote_f1 = (
            smote.get((src, tgt), {}) or {}
        )
        smote_f1 = smote_f1.get("overall", {}).get("macro_f1") if isinstance(smote_f1, dict) else None

        tqs_entry = tqs_dict.get((src, tgt), {})
        b_tqs  = tqs_entry.get("scenario_b_tqs")
        c_tqs  = tqs_entry.get("scenario_c_tqs")
        interp = tqs_entry.get("interpretation", "")
        interp = interp[:18] if interp else ""  # truncate for table width

        print(
            f"  {pair_label:<28} "
            f"{_f(scen_a.get(src)):>7}  "
            f"{_f(scen_b.get((src, tgt))):>7}  "
            f"{_f(scen_c.get((src, tgt))):>7}  "
            f"{_f(smote_f1):>7}  "
            f"{_f(base_f1.get(src)):>8}  "
            f"{_f(base_f1.get(tgt)):>8}  "
            f"{_t(b_tqs):>7}  "
            f"{_t(c_tqs):>7}  "
            f"{interp}"
        )

    print(sep_full)
    print("  Columns: A-src=Scenario A macro-F1 on source  "
          "B/C-tgt=Scenario B/C macro-F1 on target")
    print("           base-*=no-augmentation baseline  TQS=Transferability Quality Score")
    print(sep_full)

    if failed:
        print(f"\n  WARNING: {len(failed)} step(s) failed:")
        for s in failed:
            print(f"    - {s}")
    else:
        print("\n  All steps completed successfully.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # --dry_run shorthand
    if args.dry_run:
        args.synthetic = True
        args.epochs = 3
        args.n_aug  = 20
        args.projector_epochs = 20

    logger = _setup_logging()

    start_ts = datetime.now()
    logger.info("=" * 60)
    logger.info("CE-GAN Full Pipeline  started %s", start_ts.strftime("%Y-%m-%d %H:%M:%S"))
    logger.info(
        "Config: epochs=%d  batch=%d  n_aug=%d  synthetic=%s  dry_run=%s",
        args.epochs, args.batch_size, args.n_aug, args.synthetic, args.dry_run,
    )
    logger.info("=" * 60)

    # Result accumulators
    scen_a:  Dict[str, Optional[float]]         = {}
    scen_b:  Dict[Tuple, Optional[float]]        = {}
    scen_c:  Dict[Tuple, Optional[float]]        = {}
    smote_r: Dict[Tuple, Optional[Dict]]         = {}
    base_f1: Dict[str, Optional[float]]          = {}
    failed:  List[str]                           = []

    common = _common_flags(args)

    # =========================================================================
    # Phase 1: Scenario A — within-dataset baseline
    # =========================================================================
    logger.info("\n%s\nPHASE 1: Scenario A (within-dataset)\n%s", "="*60, "="*60)
    for ds in DATASETS:
        step = f"Scenario A [{ds}]"
        ok, stdout = _run_script(
            "run_scenario_a.py",
            ["--dataset", ds] + common,
            step, logger,
        )
        if ok:
            f1 = _parse_f1(stdout, r"Scenario A complete\. Macro F1: ([\d.]+)")
            scen_a[ds] = f1
            logger.info("  -> macro-F1 = %s", f"{f1:.4f}" if f1 else "parse_failed")
        else:
            failed.append(step)
            scen_a[ds] = None

    # =========================================================================
    # Phase 2: No-augmentation baselines (inline)
    # =========================================================================
    logger.info("\n%s\nPHASE 2: No-augmentation baselines\n%s", "="*60, "="*60)
    for ds in DATASETS:
        base_f1[ds] = _no_aug_f1(ds, args, logger)

    # =========================================================================
    # Phase 3: Scenario B — direct transfer
    # =========================================================================
    logger.info("\n%s\nPHASE 3: Scenario B (direct transfer)\n%s", "="*60, "="*60)
    for src, tgt in PAIRS:
        step = f"Scenario B [{src} -> {tgt}]"
        ok, stdout = _run_script(
            "run_scenario_b.py",
            ["--source", src, "--target", tgt,
             "--projector_epochs", str(args.projector_epochs)] + common,
            step, logger,
        )
        if ok:
            f1 = _parse_f1(stdout, r"Scenario B complete\. Macro F1: ([\d.]+)")
            scen_b[(src, tgt)] = f1
            logger.info("  -> macro-F1 = %s", f"{f1:.4f}" if f1 else "parse_failed")
        else:
            failed.append(step)
            scen_b[(src, tgt)] = None

    # =========================================================================
    # Phase 4: Scenario C — MMD-aligned transfer
    # =========================================================================
    logger.info("\n%s\nPHASE 4: Scenario C (MMD-aligned)\n%s", "="*60, "="*60)
    for src, tgt in PAIRS:
        step = f"Scenario C [{src} -> {tgt}]"
        ok, stdout = _run_script(
            "run_scenario_c.py",
            ["--source", src, "--target", tgt,
             "--projector_epochs", str(args.projector_epochs)] + common,
            step, logger,
        )
        if ok:
            f1 = _parse_f1(
                stdout, r"Macro F1 \(target, augmented\)\s+: ([\d.]+)"
            )
            scen_c[(src, tgt)] = f1
            logger.info("  -> macro-F1 = %s", f"{f1:.4f}" if f1 else "parse_failed")
        else:
            failed.append(step)
            scen_c[(src, tgt)] = None

    # =========================================================================
    # Phase 5: SMOTE baseline (inline)
    # =========================================================================
    if not args.skip_smote:
        logger.info("\n%s\nPHASE 5: SMOTE baseline\n%s", "="*60, "="*60)
        for src, tgt in PAIRS:
            smote_r[(src, tgt)] = _run_smote(src, tgt, args, logger)
        _save_smote_table(smote_r, logger)
    else:
        logger.info("PHASE 5: SMOTE baseline skipped (--skip_smote)")
        for pair in PAIRS:
            smote_r[pair] = None

    # =========================================================================
    # Phase 6: TQS computation
    # =========================================================================
    logger.info("\n%s\nPHASE 6: TQS computation\n%s", "="*60, "="*60)

    from src.metrics import compute_tqs, save_tqs_table

    tqs_dict: Dict = {}
    for src, tgt in PAIRS:
        f1_src_within  = scen_a.get(src)
        f1_src_base    = base_f1.get(src)
        f1_tgt_base    = base_f1.get(tgt)
        f1_b           = scen_b.get((src, tgt))
        f1_c           = scen_c.get((src, tgt))

        b_tqs, b_interp = (None, "N/A")
        c_tqs, c_interp = (None, "N/A")

        if all(v is not None for v in [f1_b, f1_tgt_base, f1_src_within, f1_src_base]):
            b_tqs, b_interp = compute_tqs(f1_b, f1_tgt_base, f1_src_within, f1_src_base)

        if all(v is not None for v in [f1_c, f1_tgt_base, f1_src_within, f1_src_base]):
            c_tqs, c_interp = compute_tqs(f1_c, f1_tgt_base, f1_src_within, f1_src_base)

        interpretation = c_interp if c_tqs is not None else b_interp

        tqs_dict[(src, tgt)] = {
            "scenario_b_tqs": b_tqs,
            "scenario_c_tqs": c_tqs,
            "interpretation": interpretation,
        }
        logger.info(
            "  TQS [%s->%s]  B=%s  C=%s  (%s)",
            src, tgt,
            f"{b_tqs:.2f}" if b_tqs is not None else "N/A",
            f"{c_tqs:.2f}" if c_tqs is not None else "N/A",
            interpretation,
        )

    save_tqs_table(tqs_dict)

    # =========================================================================
    # Phase 7: Final summary
    # =========================================================================
    elapsed = datetime.now() - start_ts
    logger.info(
        "\nPipeline finished in %dm %ds  |  %d step(s) failed",
        int(elapsed.total_seconds()) // 60,
        int(elapsed.total_seconds()) % 60,
        len(failed),
    )
    logger.info("Log file -> %s", LOG_PATH)

    _print_summary(scen_a, scen_b, scen_c, smote_r, base_f1, tqs_dict, failed)

    print(f"\nLog file -> {LOG_PATH}")
    if failed:
        print(f"Completed with {len(failed)} failure(s).")
        sys.exit(1)


if __name__ == "__main__":
    main()
