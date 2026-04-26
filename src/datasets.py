import os
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
RESULTS_TABLES = PROJECT_ROOT / "results" / "tables"


# ── Dataset-specific column config ────────────────────────────────────────────

NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty",
]
NSL_KDD_LABEL_COL = "label"
NSL_KDD_FEATURE_COLS = [c for c in NSL_KDD_COLUMNS if c not in ("label", "difficulty")]

# UNSW-NB15: official 49-column layout (no header row in raw CSV files)
UNSW_NB15_COLUMNS = [
    "srcip", "sport", "dstip", "dsport", "proto", "state",
    "dur", "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss",
    "service", "sload", "dload", "spkts", "dpkts",
    "swin", "dwin", "stcpb", "dtcpb", "smeansz", "dmeansz",
    "trans_depth", "res_bdy_len", "sjit", "djit",
    "stime", "ltime", "sintpkt", "dintpkt",
    "tcprtt", "synack", "ackdat",
    "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd",
    "is_ftp_login", "ct_ftp_cmd",
    "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "attack_cat", "label",
]
# Columns to drop from UNSW-NB15 before feature extraction
# (IP addresses are identifiers; timestamps are non-informative; binary label leaks target)
UNSW_NB15_DROP_COLS = {"srcip", "dstip", "stime", "ltime", "label"}
UNSW_NB15_LABEL_COL = "attack_cat"

# CIC-IDS2017: 78 features + label; label column is " Label" (note leading space)
CIC_IDS2017_LABEL_COL = " Label"

# Minority-class threshold: classes whose share of total samples is below this
# fraction are considered minority classes.
MINORITY_THRESHOLD = 0.10


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c != label_col]
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median(numeric_only=True))
    df[feature_cols] = df[feature_cols].fillna(0)  # fallback for all-NaN columns
    return df


def _encode_categoricals(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    for col in df.columns:
        if col == label_col:
            continue
        if df[col].dtype == object:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    return df


def _normalize(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


def _to_tensors(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(X_train, dtype=torch.float32).to(device),
        torch.tensor(X_test, dtype=torch.float32).to(device),
        torch.tensor(y_train, dtype=torch.long).to(device),
        torch.tensor(y_test, dtype=torch.long).to(device),
    )


# ── DatasetLoader ─────────────────────────────────────────────────────────────

class DatasetLoader:
    """
    Loads one of {NSL-KDD, UNSW-NB15, CIC-IDS2017}, cleans, normalises,
    and splits the data. Returns PyTorch tensors on the given device.

    Parameters
    ----------
    name : str
        One of 'nsl_kdd', 'unsw_nb15', 'cic_ids2017'.
    test_size : float
        Fraction of data reserved for the test split.
    random_state : int
        Seed for reproducibility.
    device : torch.device | None
        Target device; defaults to CUDA if available, else CPU.
    """

    SUPPORTED = ("nsl_kdd", "unsw_nb15", "cic_ids2017")

    def __init__(
        self,
        name: str,
        test_size: float = 0.2,
        random_state: int = 42,
        device: torch.device | None = None,
        max_samples: int | None = None,
    ) -> None:
        if name not in self.SUPPORTED:
            raise ValueError(f"name must be one of {self.SUPPORTED}, got '{name}'")
        self.name = name
        self.test_size = test_size
        self.random_state = random_state
        self.max_samples = max_samples
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._df: pd.DataFrame | None = None
        self._label_encoder = LabelEncoder()
        self._label_col: str = ""
        self._X_train: torch.Tensor | None = None
        self._X_test: torch.Tensor | None = None
        self._y_train: torch.Tensor | None = None
        self._y_test: torch.Tensor | None = None
        self._classes: np.ndarray | None = None

    # ── public API ────────────────────────────────────────────────────────────

    def load(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load, clean, normalise, and split the dataset.

        Returns
        -------
        X_train, X_test, y_train, y_test : torch.Tensor (on self.device)
        """
        df, label_col = self._read_csv()
        self._label_col = label_col

        df = _clean(df, label_col)
        df = _encode_categoricals(df, label_col)

        # Optional stratified down-sample for large datasets
        if self.max_samples is not None and len(df) > self.max_samples:
            from sklearn.model_selection import StratifiedShuffleSplit
            sss = StratifiedShuffleSplit(
                n_splits=1,
                train_size=self.max_samples,
                random_state=self.random_state,
            )
            idx, _ = next(sss.split(df, df[label_col]))
            df = df.iloc[idx].reset_index(drop=True)

        # Drop classes too rare to split (need at least 2 samples for train+test)
        min_cls = max(2, int(np.ceil(1 / self.test_size)) + 1)
        cls_counts = df[label_col].value_counts()
        keep_cls = cls_counts[cls_counts >= min_cls].index
        before = len(df)
        df = df[df[label_col].isin(keep_cls)].reset_index(drop=True)
        if len(df) < before:
            dropped = sorted(set(cls_counts.index) - set(keep_cls.tolist()))
            print(f"  [datasets] Dropped {len(dropped)} rare class(es) with <{min_cls} samples: {dropped}")

        feature_cols = [c for c in df.columns if c != label_col]
        X = df[feature_cols].values.astype(np.float32)
        y_raw = df[label_col].values.astype(str)

        y = self._label_encoder.fit_transform(y_raw)
        self._classes = self._label_encoder.classes_

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        X_train, X_test = _normalize(X_train, X_test)

        self._X_train, self._X_test, self._y_train, self._y_test = _to_tensors(
            X_train, X_test, y_train, y_test, self.device
        )
        self._df = df
        return self._X_train, self._X_test, self._y_train, self._y_test

    def get_class_distribution(self) -> Dict[str, int]:
        """Return {class_name: total_count} across the full (pre-split) dataset."""
        if self._df is None:
            raise RuntimeError("Call load() before get_class_distribution().")
        raw_labels = self._df[self._label_col].astype(str)
        counts = raw_labels.value_counts().to_dict()
        return {str(k): int(v) for k, v in counts.items()}

    @property
    def num_features(self) -> int:
        if self._X_train is None:
            raise RuntimeError("Call load() first.")
        return self._X_train.shape[1]

    @property
    def num_classes(self) -> int:
        if self._classes is None:
            raise RuntimeError("Call load() first.")
        return len(self._classes)

    @property
    def class_names(self) -> List[str]:
        if self._classes is None:
            raise RuntimeError("Call load() first.")
        return self._classes.tolist()

    # ── private helpers ───────────────────────────────────────────────────────

    def _read_csv(self) -> Tuple[pd.DataFrame, str]:
        folder = DATA_ROOT / self.name

        if self.name == "nsl_kdd":
            return self._load_nsl_kdd(folder)
        elif self.name == "unsw_nb15":
            return self._load_unsw_nb15(folder)
        else:
            return self._load_cic_ids2017(folder)

    def _load_nsl_kdd(self, folder: Path) -> Tuple[pd.DataFrame, str]:
        train_path = folder / "KDDTrain+.txt"
        test_path = folder / "KDDTest+.txt"

        dfs = []
        for p in (train_path, test_path):
            if not p.exists():
                raise FileNotFoundError(f"NSL-KDD file not found: {p}")
            dfs.append(pd.read_csv(p, header=None, names=NSL_KDD_COLUMNS))

        df = pd.concat(dfs, ignore_index=True)
        df = df.drop(columns=["difficulty"], errors="ignore")
        return df, NSL_KDD_LABEL_COL

    def _load_unsw_nb15(self, folder: Path) -> Tuple[pd.DataFrame, str]:
        csvs = sorted(folder.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No CSV files found in {folder}")
        # Raw CSV files have no header row — supply official column names
        dfs = [
            pd.read_csv(f, header=None, names=UNSW_NB15_COLUMNS, low_memory=False)
            for f in csvs
        ]
        df = pd.concat(dfs, ignore_index=True)

        # Drop non-feature columns (IPs, timestamps, binary label)
        df.drop(columns=[c for c in UNSW_NB15_DROP_COLS if c in df.columns], inplace=True)

        label_col = UNSW_NB15_LABEL_COL  # "attack_cat"
        if label_col not in df.columns:
            raise KeyError(
                f"Could not find label column '{label_col}' in UNSW-NB15. "
                f"Available: {df.columns.tolist()}"
            )

        df[label_col] = df[label_col].fillna("Normal").astype(str).str.strip()
        # Blank attack_cat means normal traffic
        df[label_col] = df[label_col].replace("", "Normal")
        # Normalise known label typos across CSV versions
        df[label_col] = df[label_col].replace({"Backdoors": "Backdoor"})
        return df, label_col

    def _load_cic_ids2017(self, folder: Path) -> Tuple[pd.DataFrame, str]:
        # Check root folder and any immediate subdirectory (e.g. MachineLearningCVE/)
        csvs = sorted(folder.glob("*.csv"))
        if not csvs:
            csvs = sorted(folder.glob("*/*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No CSV files found in {folder} or subdirectories")
        df = pd.concat([pd.read_csv(f, low_memory=False) for f in csvs], ignore_index=True)

        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        label_col = CIC_IDS2017_LABEL_COL.strip()

        if label_col not in df.columns:
            candidates = [c for c in df.columns if "label" in c.lower()]
            if not candidates:
                raise KeyError(
                    f"Could not find label column in CIC-IDS2017. "
                    f"Available: {df.columns.tolist()}"
                )
            label_col = candidates[0]

        df[label_col] = df[label_col].astype(str).str.strip()
        return df, label_col


# ── compute_imbalance_stats ───────────────────────────────────────────────────

def compute_imbalance_stats(loader: DatasetLoader) -> Dict:
    """
    Compute class-imbalance statistics for a loaded DatasetLoader instance.

    Returns a dict with keys:
        total_samples, num_classes, majority_class, majority_count,
        minority_classes (list of {name, count, imbalance_ratio}),
        all_classes (list of {name, count, imbalance_ratio})
    """
    dist = loader.get_class_distribution()
    total = sum(dist.values())
    sorted_dist = sorted(dist.items(), key=lambda x: x[1], reverse=True)

    majority_name, majority_count = sorted_dist[0]

    all_classes = [
        {
            "class_name": name,
            "count": count,
            "share": count / total,
            "imbalance_ratio": majority_count / count if count > 0 else float("inf"),
        }
        for name, count in sorted_dist
    ]

    minority_classes = [c for c in all_classes if c["share"] < MINORITY_THRESHOLD]

    return {
        "dataset": loader.name,
        "total_samples": total,
        "num_classes": len(dist),
        "majority_class": majority_name,
        "majority_count": majority_count,
        "minority_classes": minority_classes,
        "all_classes": all_classes,
    }


def save_imbalance_stats_csv(stats_list: List[Dict]) -> Path:
    """
    Write per-class imbalance statistics for all datasets to
    results/tables/table1_class_distribution.csv.
    """
    RESULTS_TABLES.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_TABLES / "table1_class_distribution.csv"

    fieldnames = ["dataset", "class_name", "count", "share_pct", "imbalance_ratio", "is_minority"]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for stats in stats_list:
            minority_names = {c["class_name"] for c in stats["minority_classes"]}
            for cls in stats["all_classes"]:
                writer.writerow(
                    {
                        "dataset": stats["dataset"],
                        "class_name": cls["class_name"],
                        "count": cls["count"],
                        "share_pct": f"{cls['share'] * 100:.2f}",
                        "imbalance_ratio": f"{cls['imbalance_ratio']:.2f}",
                        "is_minority": cls["class_name"] in minority_names,
                    }
                )

    return out_path


# ── __main__ ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    all_stats = []

    for ds_name in DatasetLoader.SUPPORTED:
        print(f"{'=' * 60}")
        print(f"Loading {ds_name} ...")
        try:
            loader = DatasetLoader(ds_name, device=device)
            X_train, X_test, y_train, y_test = loader.load()

            print(f"  Features : {loader.num_features}")
            print(f"  Classes  : {loader.num_classes}")
            print(f"  X_train  : {tuple(X_train.shape)}  (device={X_train.device})")
            print(f"  X_test   : {tuple(X_test.shape)}")

            stats = compute_imbalance_stats(loader)
            all_stats.append(stats)

            print(f"  Total samples : {stats['total_samples']:,}")
            print(f"  Majority class: '{stats['majority_class']}' "
                  f"({stats['majority_count']:,})")
            print(f"  Minority classes ({len(stats['minority_classes'])}):")
            for mc in stats["minority_classes"]:
                print(f"    {mc['class_name']:<30} count={mc['count']:>8,}  "
                      f"ratio={mc['imbalance_ratio']:>8.1f}x")

        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
        print()

    if all_stats:
        out = save_imbalance_stats_csv(all_stats)
        print(f"Imbalance stats saved to: {out}")
