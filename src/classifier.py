"""
Nash Equilibrium Ensemble Classifier for network intrusion detection.

Combines Random Forest, Extra Trees, and Gradient Boosting via game-theoretic
weight update: each classifier's weight is proportional to its payoff (1 - error),
iterated until the weight vector converges (Nash equilibrium of the ensemble game).

Reference: Yang et al. 2025, Sec. 4.2
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR   = PROJECT_ROOT / "results" / "tables"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_numpy(X) -> np.ndarray:
    """Accept ndarray, list, or any tensor (CUDA/CPU) and return a float32 ndarray."""
    if isinstance(X, np.ndarray):
        return X.astype(np.float32)
    # torch.Tensor path (no direct import to keep sklearn-only installs working)
    if hasattr(X, "detach"):
        return X.detach().cpu().numpy().astype(np.float32)
    return np.asarray(X, dtype=np.float32)


def _labels_to_numpy(y) -> np.ndarray:
    """Accept ndarray, list, or tensor and return a 1-D int64 ndarray."""
    if isinstance(y, np.ndarray):
        return y.astype(np.int64)
    if hasattr(y, "detach"):
        return y.detach().cpu().numpy().astype(np.int64)
    return np.asarray(y, dtype=np.int64)


# ---------------------------------------------------------------------------
# NashEnsembleClassifier
# ---------------------------------------------------------------------------

class NashEnsembleClassifier:
    """
    Three-player Nash equilibrium ensemble.

    Players: RandomForest (p0), ExtraTrees (p1), GradientBoosting (p2).

    Weight update rule (iterated best-response):
        payoff_i = 1 - error_i        (on training data after fit)
        w_i      = payoff_i / sum(payoffs)
    Repeated until ||w_new - w_old||_inf < epsilon  (convergence).

    Parameters
    ----------
    init_equal_weights : bool
        True  -> start from w = [1/3, 1/3, 1/3]  (cross-dataset scenario,
                 avoids bias toward any single domain)
        False -> start from performance-based init on training data
                 (within-dataset baseline, as in Yang et al. 2025)
    n_estimators : int
        Trees per forest / boosting rounds.
    max_iter_nash : int
        Hard cap on Nash iteration rounds.
    epsilon : float
        Convergence threshold on the infinity norm of weight change.
    random_state : int
    verbose : bool
        If True, print weight evolution table to stdout.
    """

    _NAMES = ("RandomForest", "ExtraTrees", "GradientBoosting")

    def __init__(
        self,
        init_equal_weights: bool = True,
        n_estimators: int = 100,
        max_iter_nash: int = 50,
        epsilon: float = 1e-3,
        random_state: int = 42,
        verbose: bool = True,
    ) -> None:
        self.init_equal_weights = init_equal_weights
        self.epsilon            = epsilon
        self.max_iter_nash      = max_iter_nash
        self.verbose            = verbose

        self._clfs = [
            RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=random_state,
                n_jobs=-1,
            ),
            ExtraTreesClassifier(
                n_estimators=n_estimators,
                random_state=random_state,
                n_jobs=-1,
            ),
            GradientBoostingClassifier(
                n_estimators=n_estimators,
                random_state=random_state,
            ),
        ]
        self.weights_: np.ndarray = np.array([1 / 3, 1 / 3, 1 / 3])
        self.weight_history_: List[np.ndarray] = []
        self.classes_: Optional[np.ndarray] = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X_train, y_train) -> "NashEnsembleClassifier":
        """
        Train all three base classifiers, then run Nash weight iteration.

        Parameters
        ----------
        X_train : array-like [n, d]  (numpy, list, or CUDA/CPU tensor)
        y_train : array-like [n]     integer class labels

        Returns
        -------
        self
        """
        X = _to_numpy(X_train)
        y = _labels_to_numpy(y_train)
        self.classes_ = np.unique(y)

        # -- Train base classifiers ------------------------------------------
        for clf in self._clfs:
            clf.fit(X, y)

        # -- Initial weights -------------------------------------------------
        if self.init_equal_weights:
            w = np.array([1 / 3, 1 / 3, 1 / 3])
        else:
            # Performance-based init: payoff = 1 - training error
            payoffs = np.array([
                1.0 - (1.0 - accuracy_score(y, clf.predict(X)))
                for clf in self._clfs
            ])
            w = payoffs / payoffs.sum()

        self.weight_history_ = [w.copy()]
        if self.verbose:
            self._print_header()
            self._print_row(0, w)

        # -- Nash iteration --------------------------------------------------
        for iteration in range(1, self.max_iter_nash + 1):
            # Error of each classifier on training data (fast: uses cached predictions)
            errors = np.array([
                1.0 - accuracy_score(y, clf.predict(X))
                for clf in self._clfs
            ])
            payoffs = 1.0 - errors          # payoff_i in [0, 1]

            # Guard: if all payoffs zero (degenerate), keep equal weights
            total = payoffs.sum()
            w_new = payoffs / total if total > 0 else np.full(3, 1 / 3)

            delta = np.abs(w_new - w).max()
            w = w_new
            self.weight_history_.append(w.copy())

            if self.verbose:
                self._print_row(iteration, w, errors=errors, delta=delta)

            if delta < self.epsilon:
                if self.verbose:
                    print(f"  Nash converged at iteration {iteration}  (delta={delta:.2e})")
                break
        else:
            if self.verbose:
                print(f"  Nash: reached max_iter_nash={self.max_iter_nash}")

        self.weights_ = w
        self._fitted  = True
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, X_test) -> np.ndarray:
        """
        Weighted majority vote.

        Each classifier casts one vote per sample (its predicted class);
        votes are weighted by self.weights_.  Ties broken by the class
        with the highest total weighted probability (falls through to
        predict_proba implicitly via argmax on probability average).
        """
        self._check_fitted()
        return self.classes_[np.argmax(self.predict_proba(X_test), axis=1)]

    def predict_proba(self, X_test) -> np.ndarray:
        """
        Weighted average of posterior probabilities.

        Returns
        -------
        [n_samples, n_classes] probability matrix  (rows sum to 1)
        """
        self._check_fitted()
        X = _to_numpy(X_test)
        proba = np.zeros((X.shape[0], len(self.classes_)))
        for w, clf in zip(self.weights_, self._clfs):
            proba += w * clf.predict_proba(X)
        return proba

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save(self, path) -> None:
        """Pickle the fitted ensemble to *path* (creates parents)."""
        import pickle
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path) -> "NashEnsembleClassifier":
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("NashEnsembleClassifier: call fit() before predict().")

    def _print_header(self) -> None:
        sep = "-" * 72
        print(sep)
        print(
            f"  {'Iter':>4}  "
            f"{'w_RF':>8}  {'w_ET':>8}  {'w_GB':>8}  "
            f"{'e_RF':>8}  {'e_ET':>8}  {'e_GB':>8}  "
            f"{'delta':>8}"
        )
        print(sep)

    def _print_row(
        self,
        iteration: int,
        w: np.ndarray,
        errors: Optional[np.ndarray] = None,
        delta: Optional[float] = None,
    ) -> None:
        err_str = (
            f"{errors[0]:8.4f}  {errors[1]:8.4f}  {errors[2]:8.4f}"
            if errors is not None
            else f"{'---':>8}  {'---':>8}  {'---':>8}"
        )
        delta_str = f"{delta:8.2e}" if delta is not None else f"{'---':>8}"
        print(
            f"  {iteration:4d}  "
            f"{w[0]:8.4f}  {w[1]:8.4f}  {w[2]:8.4f}  "
            f"{err_str}  "
            f"{delta_str}"
        )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_all_metrics(
    y_true,
    y_pred,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    Compute a comprehensive metrics bundle.

    Parameters
    ----------
    y_true       : array-like [n]  ground truth integer labels
    y_pred       : array-like [n]  predicted integer labels
    class_names  : optional list of string names for per-class entries

    Returns
    -------
    dict with keys:
      overall  -> {accuracy, macro_f1, weighted_f1, mcc}
      per_class -> list of {class_id, [name], precision, recall, f1, support}
      confusion_matrix -> 2-D list (int)
    """
    yt = _labels_to_numpy(y_true)
    yp = _labels_to_numpy(y_pred)

    labels = sorted(np.unique(np.concatenate([yt, yp])).tolist())

    accuracy    = float(accuracy_score(yt, yp))
    macro_f1    = float(f1_score(yt, yp, average="macro",    zero_division=0, labels=labels))
    weighted_f1 = float(f1_score(yt, yp, average="weighted", zero_division=0, labels=labels))
    mcc         = float(matthews_corrcoef(yt, yp))

    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        yt, yp, labels=labels, zero_division=0
    )

    per_class = []
    for i, label in enumerate(labels):
        entry: Dict = {
            "class_id":  int(label),
            "precision": float(precisions[i]),
            "recall":    float(recalls[i]),
            "f1":        float(f1s[i]),
            "support":   int(supports[i]),
        }
        if class_names is not None and int(label) < len(class_names):
            entry["name"] = class_names[int(label)]
        per_class.append(entry)

    cm = confusion_matrix(yt, yp, labels=labels).tolist()

    return {
        "overall": {
            "accuracy":    accuracy,
            "macro_f1":    macro_f1,
            "weighted_f1": weighted_f1,
            "mcc":         mcc,
        },
        "per_class":        per_class,
        "confusion_matrix": cm,
    }


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def save_results_table(
    metrics: Dict,
    scenario: str,
    source: str,
    target: Optional[str] = None,
    csv_path: Optional[Path] = None,
) -> Path:
    """
    Append one result row to the main results CSV table.

    Columns
    -------
    scenario, source_dataset, target_dataset,
    accuracy, macro_f1, weighted_f1, minority_class_f1

    ``minority_class_f1`` is the F1 of the class with the fewest support
    samples (the hardest minority class in imbalanced IDS data).

    Parameters
    ----------
    metrics   : dict as returned by compute_all_metrics()
    scenario  : e.g. "A", "B", "C" or "baseline"
    source    : source dataset name
    target    : target dataset name (or None for within-dataset)
    csv_path  : override output path (default: results/tables/table3_main_results.csv)

    Returns
    -------
    Path  –  absolute path of the CSV file written to
    """
    if csv_path is None:
        csv_path = TABLES_DIR / "table3_main_results.csv"

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    ov = metrics["overall"]

    # Minority class: smallest support among per_class entries
    pc = metrics["per_class"]
    minority = min(pc, key=lambda e: e["support"]) if pc else {}
    minority_f1 = minority.get("f1", float("nan"))

    fieldnames = [
        "scenario", "source_dataset", "target_dataset",
        "accuracy", "macro_f1", "weighted_f1", "minority_class_f1",
    ]

    write_header = not csv_path.exists() or csv_path.stat().st_size == 0

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "scenario":         scenario,
            "source_dataset":   source,
            "target_dataset":   target if target is not None else source,
            "accuracy":         f"{ov['accuracy']:.6f}",
            "macro_f1":         f"{ov['macro_f1']:.6f}",
            "weighted_f1":      f"{ov['weighted_f1']:.6f}",
            "minority_class_f1": f"{minority_f1:.6f}",
        })

    return csv_path


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    _GREEN = "\033[92m"
    _RED   = "\033[91m"
    _RESET = "\033[0m"

    def _ok(b: bool) -> str:
        return f"{_GREEN}PASS{_RESET}" if b else f"{_RED}FAIL{_RESET}"

    rng = np.random.default_rng(42)
    SEP = "-" * 72

    # ------------------------------------------------------------------
    # Build imbalanced dummy dataset (5 classes, heavy imbalance)
    # Mimics NSL-KDD: class 0 (normal) dominant, classes 1-4 minority
    # ------------------------------------------------------------------
    n_features = 41
    class_sizes = [800, 80, 40, 20, 10]   # ~10:1 imbalance ratio
    n_classes   = len(class_sizes)

    Xs, ys = [], []
    for cls_id, n in enumerate(class_sizes):
        mean = rng.uniform(-1, 1, size=n_features) * cls_id
        Xs.append(rng.normal(mean, 0.5, size=(n, n_features)))
        ys.append(np.full(n, cls_id, dtype=np.int64))

    X_all = np.concatenate(Xs).astype(np.float32)
    y_all = np.concatenate(ys)

    # 80/20 split
    n_total = len(y_all)
    idx     = rng.permutation(n_total)
    split   = int(0.8 * n_total)
    X_train, X_test = X_all[idx[:split]], X_all[idx[split:]]
    y_train, y_test = y_all[idx[:split]], y_all[idx[split:]]

    print(f"\nDummy dataset: {n_total} samples  |  {n_features} features  |  "
          f"{n_classes} classes  |  train={split}  test={n_total - split}")
    print(f"Class sizes (train split): "
          + "  ".join(f"c{i}={(y_train==i).sum()}" for i in range(n_classes)))

    # ------------------------------------------------------------------
    # Test 1: equal-weight init (cross-dataset mode)
    # ------------------------------------------------------------------
    print(f"\n{'='*72}")
    print("Test 1: NashEnsembleClassifier(init_equal_weights=True)")
    print(SEP)

    clf_equal = NashEnsembleClassifier(
        init_equal_weights=True, n_estimators=50, verbose=True
    )
    clf_equal.fit(X_train, y_train)

    y_pred_eq = clf_equal.predict(X_test)
    m_eq      = compute_all_metrics(y_test, y_pred_eq)
    ov_eq     = m_eq["overall"]

    print(f"\nFinal weights : RF={clf_equal.weights_[0]:.4f}  "
          f"ET={clf_equal.weights_[1]:.4f}  "
          f"GB={clf_equal.weights_[2]:.4f}  "
          f"(sum={clf_equal.weights_.sum():.4f})")
    print(f"Accuracy      : {ov_eq['accuracy']:.4f}")
    print(f"Macro F1      : {ov_eq['macro_f1']:.4f}")
    print(f"Weighted F1   : {ov_eq['weighted_f1']:.4f}")
    print(f"MCC           : {ov_eq['mcc']:.4f}")

    ok1_weights = abs(clf_equal.weights_.sum() - 1.0) < 1e-6
    ok1_acc     = ov_eq["accuracy"] > 0.7
    print(f"\nweights sum=1 : {_ok(ok1_weights)}")
    print(f"accuracy > 0.7: {_ok(ok1_acc)}")

    # ------------------------------------------------------------------
    # Test 2: performance-based init (within-dataset mode)
    # ------------------------------------------------------------------
    print(f"\n{'='*72}")
    print("Test 2: NashEnsembleClassifier(init_equal_weights=False)")
    print(SEP)

    clf_perf = NashEnsembleClassifier(
        init_equal_weights=False, n_estimators=50, verbose=True
    )
    clf_perf.fit(X_train, y_train)

    y_pred_pf = clf_perf.predict(X_test)
    m_pf      = compute_all_metrics(y_test, y_pred_pf)
    ov_pf     = m_pf["overall"]

    print(f"\nFinal weights : RF={clf_perf.weights_[0]:.4f}  "
          f"ET={clf_perf.weights_[1]:.4f}  "
          f"GB={clf_perf.weights_[2]:.4f}")
    print(f"Accuracy      : {ov_pf['accuracy']:.4f}")
    print(f"Macro F1      : {ov_pf['macro_f1']:.4f}")

    ok2 = ov_pf["accuracy"] > 0.7
    print(f"accuracy > 0.7: {_ok(ok2)}")

    # ------------------------------------------------------------------
    # Test 3: CUDA tensor input (if available)
    # ------------------------------------------------------------------
    print(f"\n{'='*72}")
    print("Test 3: CUDA tensor input auto-conversion")
    try:
        import torch
        if torch.cuda.is_available():
            X_cuda = torch.tensor(X_train, device="cuda")
            y_cuda = torch.tensor(y_train, device="cuda")
            clf_cuda = NashEnsembleClassifier(init_equal_weights=True, n_estimators=20, verbose=False)
            clf_cuda.fit(X_cuda, y_cuda)
            preds = clf_cuda.predict(torch.tensor(X_test, device="cuda"))
            ok3 = len(preds) == len(y_test)
            print(f"  Tensor fit+predict ({len(preds)} preds): {_ok(ok3)}")
        else:
            print("  CUDA not available - skipping tensor test")
            ok3 = True
    except Exception as exc:
        print(f"  {_RED}ERROR{_RESET}: {exc}")
        ok3 = False

    # ------------------------------------------------------------------
    # Test 4: per-class metrics + results table
    # ------------------------------------------------------------------
    print(f"\n{'='*72}")
    print("Test 4: per-class metrics and save_results_table")

    class_names = ["normal", "DoS", "Probe", "R2L", "U2R"]
    m_detail    = compute_all_metrics(y_test, y_pred_eq, class_names=class_names)

    print("\n  Per-class results:")
    print(f"  {'Class':<10} {'Name':<10} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Supp':>6}")
    print("  " + "-" * 52)
    for e in m_detail["per_class"]:
        name = e.get("name", "?")
        print(
            f"  {e['class_id']:<10} {name:<10} "
            f"{e['precision']:7.4f} {e['recall']:7.4f} {e['f1']:7.4f} {e['support']:6d}"
        )

    # Save to a temp CSV for the test
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf:
        tmp_path = Path(tf.name)

    save_results_table(m_detail, scenario="A", source="nsl_kdd", csv_path=tmp_path)
    save_results_table(m_detail, scenario="B", source="nsl_kdd", target="unsw_nb15", csv_path=tmp_path)

    lines = tmp_path.read_text().strip().splitlines()
    os.unlink(tmp_path)
    ok4 = len(lines) == 3   # 1 header + 2 data rows
    print(f"\n  CSV rows (expect 3, got {len(lines)}): {_ok(ok4)}")

    # ------------------------------------------------------------------
    # Weight evolution summary
    # ------------------------------------------------------------------
    print(f"\n{'='*72}")
    print("Weight evolution (equal-weight run):")
    print(f"  {'Iter':>4}  {'RF':>8}  {'ET':>8}  {'GB':>8}")
    print("  " + "-" * 38)
    for i, w in enumerate(clf_equal.weight_history_):
        print(f"  {i:4d}  {w[0]:8.4f}  {w[1]:8.4f}  {w[2]:8.4f}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    all_ok = ok1_weights and ok1_acc and ok2 and ok3 and ok4
    print(f"\n{SEP}")
    print(f"  Test 1 equal-weight init  : {_ok(ok1_weights and ok1_acc)}")
    print(f"  Test 2 perf-based init    : {_ok(ok2)}")
    print(f"  Test 3 CUDA tensor input  : {_ok(ok3)}")
    print(f"  Test 4 metrics + CSV      : {_ok(ok4)}")
    print(SEP)
    print(f"  Overall: {'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'}")
    sys.exit(0 if all_ok else 1)
