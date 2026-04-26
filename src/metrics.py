"""
Evaluation metrics for cross-dataset CE-GAN transferability.

Implements the Transferability Quality Score (TQS) from Yang et al. 2025:

    TQS = (F1_target_aug - F1_target_base)
          / (F1_source_within - F1_source_base)
          * 100

Interpretation thresholds:
    TQS >= 80  : Excellent transfer
    50 <= TQS < 80 : Good transfer
    0  <= TQS < 50 : Partial transfer
    TQS < 0        : Negative transfer - harmful
"""

import csv
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR   = PROJECT_ROOT / "results" / "tables"


# ---------------------------------------------------------------------------
# TQS
# ---------------------------------------------------------------------------

def compute_tqs(
    f1_target_augmented: float,
    f1_target_baseline:  float,
    f1_source_within:    float,
    f1_source_baseline:  float,
) -> Tuple[Optional[float], str]:
    """
    Compute the Transferability Quality Score (TQS).

    Parameters
    ----------
    f1_target_augmented : macro-F1 on target test set WITH CE-GAN augmentation
    f1_target_baseline  : macro-F1 on target test set WITHOUT augmentation
    f1_source_within    : macro-F1 on source test set WITH within-dataset CE-GAN
    f1_source_baseline  : macro-F1 on source test set WITHOUT augmentation

    Returns
    -------
    (tqs_value, interpretation)
    tqs_value is None when the denominator is ~0 (undefined).
    """
    denom = f1_source_within - f1_source_baseline
    if abs(denom) < 1e-9:
        warnings.warn(
            "TQS denominator (F1_source_within - F1_source_base) is ~0. "
            "Within-dataset CE-GAN augmentation had no measurable effect.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None, "Undefined (denominator ~0)"

    tqs = (f1_target_augmented - f1_target_baseline) / denom * 100.0

    if tqs >= 80.0:
        interpretation = "Excellent transfer"
    elif tqs >= 50.0:
        interpretation = "Good transfer"
    elif tqs >= 0.0:
        interpretation = "Partial transfer"
    else:
        interpretation = "Negative transfer - harmful"

    return tqs, interpretation


# ---------------------------------------------------------------------------
# TQS results table
# ---------------------------------------------------------------------------

def save_tqs_table(
    results_dict: Dict,
    csv_path: Optional[Path] = None,
) -> Path:
    """
    Write the pairwise TQS results table.

    Parameters
    ----------
    results_dict : {(source, target): {
                      "scenario_b_tqs": float | None,
                      "scenario_c_tqs": float | None,
                      "interpretation": str,
                  }}
    csv_path     : override default path (default: results/tables/table4_tqs_scores.csv)

    Returns
    -------
    Path of the CSV file written
    """
    if csv_path is None:
        csv_path = TABLES_DIR / "table4_tqs_scores.csv"

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "source", "target",
        "scenario_b_tqs", "scenario_c_tqs",
        "interpretation",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (src, tgt), vals in results_dict.items():
            b = vals.get("scenario_b_tqs")
            c = vals.get("scenario_c_tqs")
            writer.writerow({
                "source":         src,
                "target":         tgt,
                "scenario_b_tqs": f"{b:.2f}" if b is not None else "N/A",
                "scenario_c_tqs": f"{c:.2f}" if c is not None else "N/A",
                "interpretation": vals.get("interpretation", ""),
            })

    print(f"TQS table saved -> {csv_path}")
    return csv_path


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    _G = "\033[92m"
    _R = "\033[91m"
    _E = "\033[0m"

    def _ok(b: bool) -> str:
        return f"{_G}PASS{_E}" if b else f"{_R}FAIL{_E}"

    SEP = "-" * 56
    print(SEP)

    # Test 1: Excellent transfer
    tqs, interp = compute_tqs(0.85, 0.60, 0.90, 0.65)
    ok1 = interp == "Excellent transfer" and abs(tqs - 100.0) < 0.1
    print(f"Test 1 (Excellent): TQS={tqs:.1f}  '{interp}'  -> {_ok(ok1)}")

    # Test 2: Good transfer   (0.12 / 0.20 * 100 = 60.0)
    tqs, interp = compute_tqs(0.72, 0.60, 0.80, 0.60)
    ok2 = interp == "Good transfer" and abs(tqs - 60.0) < 0.1
    print(f"Test 2 (Good):      TQS={tqs:.1f}  '{interp}'  -> {_ok(ok2)}")

    # Test 3: Partial transfer
    tqs, interp = compute_tqs(0.64, 0.60, 0.90, 0.65)
    ok3 = interp == "Partial transfer"
    print(f"Test 3 (Partial):   TQS={tqs:.1f}  '{interp}'  -> {_ok(ok3)}")

    # Test 4: Negative transfer
    tqs, interp = compute_tqs(0.58, 0.60, 0.90, 0.65)
    ok4 = interp == "Negative transfer - harmful"
    print(f"Test 4 (Negative):  TQS={tqs:.1f}  '{interp}'  -> {_ok(ok4)}")

    # Test 5: Zero denominator
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        tqs5, interp5 = compute_tqs(0.80, 0.70, 0.75, 0.75)
    ok5 = tqs5 is None and len(w) == 1
    print(f"Test 5 (Zero denom): tqs={tqs5}  '{interp5}'  -> {_ok(ok5)}")

    # Test 6: save_tqs_table
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf:
        tmp = Path(tf.name)
    save_tqs_table(
        {
            ("nsl_kdd", "unsw_nb15"):   {"scenario_b_tqs": 45.2, "scenario_c_tqs": 73.8, "interpretation": "Good transfer"},
            ("nsl_kdd", "cic_ids2017"): {"scenario_b_tqs": None, "scenario_c_tqs": 81.0, "interpretation": "Excellent transfer"},
        },
        csv_path=tmp,
    )
    lines = tmp.read_text().strip().splitlines()
    os.unlink(tmp)
    ok6 = len(lines) == 3   # header + 2 rows
    print(f"Test 6 (CSV rows):  got {len(lines)}, want 3  -> {_ok(ok6)}")

    print(SEP)
    all_ok = ok1 and ok2 and ok3 and ok4 and ok5 and ok6
    print(f"  Overall: {'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'}")
    sys.exit(0 if all_ok else 1)
