"""
Cross-dataset feature harmonization for CE-GAN transfer.

Step 1 – Semantic mapping: manually identified feature equivalents are
         copied directly (no information loss).
Step 2 – FC projection: a lightweight Linear->ReLU->Linear network,
         trained on target data, fills in every target feature that has
         no direct source counterpart.

References
----------
NSL-KDD  : Tavallaee et al. 2009  (41 traffic features)
UNSW-NB15: Moustafa & Slay 2015  (49 network flow features)
CIC-IDS17: Sharafaldin et al. 2018 (78 CICFlowMeter features)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_TABLES = PROJECT_ROOT / "results" / "tables"
RESULTS_FIGURES = PROJECT_ROOT / "results" / "figures"

# ── Known feature columns per dataset ─────────────────────────────────────────
# Used by the __main__ demo when real CSVs are absent.

NSL_KDD_FEATURES: List[str] = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]

UNSW_NB15_FEATURES: List[str] = [
    "dur", "proto", "service", "state", "spkts", "dpkts", "sbytes", "dbytes",
    "rate", "sttl", "dttl", "sload", "dload", "sloss", "dloss", "sinpkt",
    "dinpkt", "sjit", "djit", "swin", "stcpb", "dtcpb", "dwin", "tcprtt",
    "synack", "ackdat", "smean", "dmean", "trans_depth", "response_body_len",
    "ct_srv_src", "ct_state_ttl", "ct_dst_ltm", "ct_src_dport_ltm",
    "ct_dst_sport_ltm", "ct_dst_src_ltm", "is_ftp_login", "ct_ftp_cmd",
    "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports",
    # Padding to reach 49 – may vary by CSV version
    "stcpb_ext", "dtcpb_ext", "smeansz", "dmeansz", "res_bdy_len",
    "ct_src_src_ltm", "ct_dst_ltm_ext",
]

CIC_IDS2017_FEATURES: List[str] = [
    "Destination Port", "Flow Duration", "Total Fwd Packets",
    "Total Backward Packets", "Total Length of Fwd Packets",
    "Total Length of Bwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max",
    "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std",
    "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags",
    "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length", "Max Packet Length",
    "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count",
    "ACK Flag Count", "URG Flag Count", "CWE Flag Count", "ECE Flag Count",
    "Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size",
    "Avg Bwd Segment Size", "Fwd Header Length.1",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets",
    "Subflow Bwd Bytes", "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "act_data_pkt_fwd", "min_seg_size_forward",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]

DATASET_FEATURES: Dict[str, List[str]] = {
    "nsl_kdd":      NSL_KDD_FEATURES,
    "unsw_nb15":    UNSW_NB15_FEATURES,
    "cic_ids2017":  CIC_IDS2017_FEATURES,
}

# ── Semantic feature mapping ───────────────────────────────────────────────────
# Each entry: (source_feature, target_feature, mapping_type)
# mapping_type: "identical" | "semantic" | "projected"
# Rules: each source feature appears at most once; each target feature at most once.
# "identical" = exact same name and meaning
# "semantic"  = different name, same underlying network property
# "projected" = rough proxy; FC projector learns a data-driven refinement

_MapList = List[Tuple[str, str, str]]

FEATURE_MAPPING: Dict[Tuple[str, str], _MapList] = {

    # ── NSL-KDD (41) --> UNSW-NB15 (49) ───────────────────────────────────────
    ("nsl_kdd", "unsw_nb15"): [
        # --- temporal ---
        ("duration",                   "dur",               "semantic"),
        # --- protocol / service ---
        ("protocol_type",              "proto",             "semantic"),
        ("service",                    "service",           "identical"),
        ("flag",                       "state",             "semantic"),
        # --- byte counts ---
        ("src_bytes",                  "sbytes",            "semantic"),
        ("dst_bytes",                  "dbytes",            "semantic"),
        # --- packet anomalies ---
        ("wrong_fragment",             "sloss",             "semantic"),
        ("land",                       "is_sm_ips_ports",   "semantic"),
        ("urgent",                     "ct_ftp_cmd",        "semantic"),
        # --- connection-quality rates ---
        ("serror_rate",                "synack",            "semantic"),
        ("rerror_rate",                "tcprtt",            "semantic"),
        ("srv_serror_rate",            "sjit",              "semantic"),
        ("srv_rerror_rate",            "djit",              "semantic"),
        # --- host-level connection counts ---
        ("count",                      "ct_srv_src",        "semantic"),
        ("srv_count",                  "ct_srv_dst",        "semantic"),
        ("dst_host_count",             "ct_dst_ltm",        "semantic"),
        ("dst_host_srv_count",         "ct_src_ltm",        "semantic"),
        ("dst_host_same_src_port_rate","ct_src_dport_ltm",  "semantic"),
        ("dst_host_srv_diff_host_rate","ct_dst_sport_ltm",  "semantic"),
        # --- proportional rates ---
        ("same_srv_rate",              "rate",              "semantic"),
        ("diff_srv_rate",              "ct_state_ttl",      "semantic"),
        ("hot",                        "ct_flw_http_mthd",  "semantic"),
        # --- login / session ---
        ("logged_in",                  "is_ftp_login",      "semantic"),
        # --- projected (rough proxies) ---
        ("num_failed_logins",          "trans_depth",       "projected"),
        ("num_compromised",            "response_body_len", "projected"),
        ("num_root",                   "sttl",              "projected"),
        ("num_file_creations",         "dttl",              "projected"),
        ("num_shells",                 "sload",             "projected"),
        ("num_access_files",           "dload",             "projected"),
        ("root_shell",                 "swin",              "projected"),
        ("su_attempted",               "dwin",              "projected"),
        ("dst_host_same_srv_rate",     "smean",             "projected"),
        ("dst_host_diff_srv_rate",     "dmean",             "projected"),
        ("dst_host_serror_rate",       "stcpb",             "projected"),
        ("dst_host_srv_serror_rate",   "dtcpb",             "projected"),
        ("dst_host_rerror_rate",       "sinpkt",            "projected"),
        ("dst_host_srv_rerror_rate",   "dinpkt",            "projected"),
        ("srv_diff_host_rate",         "ackdat",            "projected"),
        ("is_host_login",              "spkts",             "projected"),
        ("is_guest_login",             "dpkts",             "projected"),
    ],

    # ── NSL-KDD (41) --> CIC-IDS2017 (78) ─────────────────────────────────────
    ("nsl_kdd", "cic_ids2017"): [
        # --- temporal ---
        ("duration",                   "Flow Duration",               "semantic"),
        # --- byte / packet volume ---
        ("src_bytes",                  "Total Length of Fwd Packets", "semantic"),
        ("dst_bytes",                  "Total Length of Bwd Packets", "semantic"),
        ("count",                      "Total Fwd Packets",           "semantic"),
        ("srv_count",                  "Total Backward Packets",      "semantic"),
        # --- TCP flag counters ---
        ("urgent",                     "URG Flag Count",              "semantic"),
        ("serror_rate",                "SYN Flag Count",              "semantic"),
        ("rerror_rate",                "RST Flag Count",              "semantic"),
        ("logged_in",                  "ACK Flag Count",              "semantic"),
        ("hot",                        "PSH Flag Count",              "semantic"),
        ("wrong_fragment",             "FIN Flag Count",              "semantic"),
        ("land",                       "ECE Flag Count",              "semantic"),
        # --- flow rates ---
        ("same_srv_rate",              "Fwd Packets/s",               "semantic"),
        ("diff_srv_rate",              "Bwd Packets/s",               "semantic"),
        ("dst_host_same_srv_rate",     "Flow Packets/s",              "semantic"),
        ("dst_host_diff_srv_rate",     "Flow Bytes/s",                "semantic"),
        # --- subflow counts ---
        ("dst_host_count",             "Subflow Fwd Packets",         "semantic"),
        ("dst_host_srv_count",         "Subflow Bwd Packets",         "semantic"),
        ("srv_diff_host_rate",         "Subflow Fwd Bytes",           "semantic"),
        ("dst_host_srv_diff_host_rate","Subflow Bwd Bytes",           "semantic"),
        # --- projected ---
        ("num_root",                   "Init_Win_bytes_forward",      "projected"),
        ("root_shell",                 "Init_Win_bytes_backward",     "projected"),
        ("su_attempted",               "act_data_pkt_fwd",            "projected"),
        ("num_failed_logins",          "min_seg_size_forward",        "projected"),
        ("srv_serror_rate",            "Flow IAT Std",                "projected"),
        ("srv_rerror_rate",            "Flow IAT Min",                "projected"),
        ("num_compromised",            "Packet Length Mean",          "projected"),
        ("num_shells",                 "Average Packet Size",         "projected"),
        ("num_access_files",           "Avg Fwd Segment Size",        "projected"),
        ("num_file_creations",         "Avg Bwd Segment Size",        "projected"),
        ("dst_host_same_src_port_rate","Active Mean",                 "projected"),
        ("dst_host_srv_serror_rate",   "Idle Mean",                   "projected"),
        ("dst_host_serror_rate",       "Flow IAT Mean",               "projected"),
        ("dst_host_rerror_rate",       "Flow IAT Max",                "projected"),
        ("is_host_login",              "Fwd Packet Length Mean",      "projected"),
        ("is_guest_login",             "Bwd Packet Length Mean",      "projected"),
        ("num_outbound_cmds",          "Down/Up Ratio",               "projected"),
        ("dst_host_srv_rerror_rate",   "Fwd Header Length",           "projected"),
        ("dst_host_srv_diff_host_rate","Bwd Header Length",           "projected"),
    ],

    # ── UNSW-NB15 (49) --> CIC-IDS2017 (78) ───────────────────────────────────
    ("unsw_nb15", "cic_ids2017"): [
        # --- temporal ---
        ("dur",             "Flow Duration",                "semantic"),
        # --- byte / packet volume ---
        ("sbytes",          "Total Length of Fwd Packets",  "semantic"),
        ("dbytes",          "Total Length of Bwd Packets",  "semantic"),
        ("spkts",           "Total Fwd Packets",            "semantic"),
        ("dpkts",           "Total Backward Packets",       "semantic"),
        # --- flow rates ---
        ("sload",           "Flow Bytes/s",                 "semantic"),
        ("rate",            "Flow Packets/s",               "semantic"),
        ("sload",           "Fwd Packets/s",                "semantic"),   # sload used twice – dedup in fit
        ("dload",           "Bwd Packets/s",                "semantic"),
        # --- timing / jitter ---
        ("sjit",            "Flow IAT Std",                 "semantic"),
        ("djit",            "Bwd IAT Std",                  "semantic"),
        ("sinpkt",          "Fwd IAT Mean",                 "semantic"),
        ("dinpkt",          "Bwd IAT Mean",                 "semantic"),
        ("tcprtt",          "Flow IAT Mean",                "semantic"),
        # --- TCP handshake ---
        ("synack",          "SYN Flag Count",               "semantic"),
        ("ackdat",          "ACK Flag Count",               "semantic"),
        # --- TCP window ---
        ("swin",            "Init_Win_bytes_forward",       "semantic"),
        ("dwin",            "Init_Win_bytes_backward",      "semantic"),
        # --- mean packet size ---
        ("smean",           "Avg Fwd Segment Size",         "semantic"),
        ("dmean",           "Avg Bwd Segment Size",         "semantic"),
        # --- connection counts ---
        ("ct_srv_src",      "Subflow Fwd Packets",          "semantic"),
        ("ct_srv_dst",      "Subflow Bwd Packets",          "semantic"),
        ("response_body_len","Subflow Fwd Bytes",           "semantic"),
        # --- projected ---
        ("sttl",            "Fwd Header Length",            "projected"),
        ("dttl",            "Bwd Header Length",            "projected"),
        ("sloss",           "FIN Flag Count",               "projected"),
        ("dloss",           "RST Flag Count",               "projected"),
        ("is_ftp_login",    "Min Packet Length",            "projected"),
        ("ct_ftp_cmd",      "URG Flag Count",               "projected"),
        ("is_sm_ips_ports", "ECE Flag Count",               "projected"),
        ("ct_state_ttl",    "PSH Flag Count",               "projected"),
        ("ct_dst_ltm",      "Active Mean",                  "projected"),
        ("ct_src_ltm",      "Idle Mean",                    "projected"),
        ("ct_dst_src_ltm",  "Active Std",                   "projected"),
        ("ct_src_dport_ltm","Idle Std",                     "projected"),
        ("trans_depth",     "act_data_pkt_fwd",             "projected"),
        ("ct_flw_http_mthd","Down/Up Ratio",                "projected"),
        ("stcpb",           "Fwd Avg Bytes/Bulk",           "projected"),
        ("dtcpb",           "Bwd Avg Bytes/Bulk",           "projected"),
    ],
}


# ── FC projector ───────────────────────────────────────────────────────────────

class _FeatureProjector(nn.Module):
    """Lightweight FC network: maps n_in mapped features -> n_out missing features."""

    def __init__(self, n_in: int, n_out: int) -> None:
        super().__init__()
        d_h = max(n_in * 2, n_out, 64)
        self.net = nn.Sequential(
            nn.Linear(n_in, d_h),
            nn.ReLU(),
            nn.Linear(d_h, n_out),
            nn.Sigmoid(),   # clamp output to [0,1] – matches min-max normalised scale
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── FeatureHarmonizer ──────────────────────────────────────────────────────────

class FeatureHarmonizer:
    """
    Maps source-dataset samples into the target dataset's feature space.

    Workflow
    --------
    1. ``fit(source_df, target_df)``  – resolve mapping, train projector
    2. ``transform(source_tensor)``   – copy semantic features, project the rest
    """

    def __init__(self, source_name: str, target_name: str) -> None:
        src, tgt = source_name.lower(), target_name.lower()
        if (src, tgt) in FEATURE_MAPPING:
            raw = FEATURE_MAPPING[(src, tgt)]
            self._reversed = False
        elif (tgt, src) in FEATURE_MAPPING:
            raw = [(t, s, mt) for s, t, mt in FEATURE_MAPPING[(tgt, src)]]
            self._reversed = True
        else:
            raise ValueError(
                f"No mapping defined for '{source_name}' -> '{target_name}'. "
                f"Available pairs: {list(FEATURE_MAPPING.keys())}"
            )
        self.source_name = src
        self.target_name = tgt
        self._raw_mapping: _MapList = raw

        # Set after fit()
        self.mapped_pairs: List[Tuple[str, str, str]] = []  # (src_col, tgt_col, type)
        self._src_idx: List[int] = []
        self._tgt_idx: List[int] = []
        self._unmapped_tgt_idx: List[int] = []
        self._projector: Optional[_FeatureProjector] = None
        self._n_target: int = 0
        self.source_cols: List[str] = []
        self.target_cols: List[str] = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── fit ────────────────────────────────────────────────────────────────────

    def fit(
        self,
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
        projector_epochs: int = 200,
    ) -> "FeatureHarmonizer":
        """
        Resolve the semantic mapping against the actual DataFrame columns,
        then train the FC projector on target data to fill missing features.

        Parameters
        ----------
        source_df        : raw source DataFrame (column names must match mapping)
        target_df        : raw target DataFrame
        projector_epochs : gradient steps for projector training (default 200)
        """
        src_cols = self._feature_cols(source_df, self.source_name)
        tgt_cols = self._feature_cols(target_df, self.target_name)
        self.source_cols = src_cols
        self.target_cols = tgt_cols
        self._n_target = len(tgt_cols)

        # Deduplicate: each source and each target feature used at most once
        seen_src: set = set()
        seen_tgt: set = set()
        self.mapped_pairs = []
        for s, t, mt in self._raw_mapping:
            if s in src_cols and t in tgt_cols and s not in seen_src and t not in seen_tgt:
                self.mapped_pairs.append((s, t, mt))
                seen_src.add(s)
                seen_tgt.add(t)

        self._src_idx = [src_cols.index(s) for s, _, _ in self.mapped_pairs]
        self._tgt_idx = [tgt_cols.index(t) for _, t, _ in self.mapped_pairs]

        mapped_tgt_set = set(self._tgt_idx)
        self._unmapped_tgt_idx = [i for i in range(len(tgt_cols)) if i not in mapped_tgt_set]

        # Train the projector on target data
        if self._unmapped_tgt_idx:
            self._train_projector(target_df, tgt_cols, projector_epochs)

        return self

    def _train_projector(
        self,
        target_df: pd.DataFrame,
        tgt_cols: List[str],
        epochs: int,
    ) -> None:
        """Self-supervised: predict unmapped target features from mapped ones."""
        numeric = target_df[tgt_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

        # Min-max normalise per column inside target
        arr = numeric.values.astype(np.float32)
        col_min = arr.min(0, keepdims=True)
        col_max = arr.max(0, keepdims=True)
        denom = np.where((col_max - col_min) > 0, col_max - col_min, 1.0)
        arr = (arr - col_min) / denom

        X_map = torch.tensor(arr[:, self._tgt_idx], dtype=torch.float32).to(self.device)
        X_unmap = torch.tensor(
            arr[:, self._unmapped_tgt_idx], dtype=torch.float32
        ).to(self.device)

        n_in  = len(self._tgt_idx)
        n_out = len(self._unmapped_tgt_idx)
        self._projector = _FeatureProjector(n_in, n_out).to(self.device)

        opt = Adam(self._projector.parameters(), lr=1e-3)
        batch = min(512, X_map.size(0))
        for epoch in range(epochs):
            idx = torch.randperm(X_map.size(0), device=self.device)[:batch]
            opt.zero_grad()
            loss = F.mse_loss(self._projector(X_map[idx]), X_unmap[idx])
            loss.backward()
            opt.step()

    # ── transform ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def transform(self, source_tensor: torch.Tensor) -> torch.Tensor:
        """
        Map source samples into the target feature space.

        Parameters
        ----------
        source_tensor : [N, n_source_features]  on CUDA, min-max normalised

        Returns
        -------
        [N, n_target_features]  on CUDA, values in [0, 1]
        """
        if not self.mapped_pairs:
            raise RuntimeError("Call fit() before transform().")

        N = source_tensor.size(0)
        out = torch.zeros(N, self._n_target, device=self.device)

        # Step 1 – copy directly mapped features
        for s_i, t_i in zip(self._src_idx, self._tgt_idx):
            out[:, t_i] = source_tensor[:, s_i]

        # Step 2 – project unmapped target features
        if self._unmapped_tgt_idx and self._projector is not None:
            self._projector.eval()
            mapped_vals = out[:, self._tgt_idx]       # already filled in step 1
            proj = self._projector(mapped_vals)        # [N, n_unmapped]
            for local_i, global_i in enumerate(self._unmapped_tgt_idx):
                out[:, global_i] = proj[:, local_i]

        return out.clamp(0.0, 1.0)

    # ── summary ────────────────────────────────────────────────────────────────

    def summary(self) -> Dict:
        n_semantic  = sum(1 for _, _, t in self.mapped_pairs if t in ("identical", "semantic"))
        n_projected = sum(1 for _, _, t in self.mapped_pairs if t == "projected")
        n_unmapped  = len(self._unmapped_tgt_idx)
        return {
            "source":          self.source_name,
            "target":          self.target_name,
            "n_source_feat":   len(self.source_cols),
            "n_target_feat":   self._n_target,
            "n_semantic":      n_semantic,
            "n_projected_map": n_projected,
            "n_fc_projected":  n_unmapped,
            "n_total_covered": len(self.mapped_pairs) + n_unmapped,
        }

    # ── helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _feature_cols(df: pd.DataFrame, dataset_name: str) -> List[str]:
        """Return numeric feature columns, excluding known label columns."""
        label_keywords = {"label", "attack_cat", "attack", "class", "target"}
        return [
            c for c in df.columns
            if c.lower() not in label_keywords
        ]


# ── Save mapping table ─────────────────────────────────────────────────────────

def save_mapping_table(
    source_name: str,
    target_name: str,
    harmonizer: Optional[FeatureHarmonizer] = None,
) -> Path:
    """
    Write the full feature mapping to
    ``results/tables/table2_feature_mapping.csv``.

    If a fitted harmonizer is provided, only rows for features that were
    actually found in the data are marked as active.
    """
    RESULTS_TABLES.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_TABLES / "table2_feature_mapping.csv"

    key = (source_name.lower(), target_name.lower())
    rev = (target_name.lower(), source_name.lower())

    if key in FEATURE_MAPPING:
        rows = [(s, t, mt, source_name, target_name)
                for s, t, mt in FEATURE_MAPPING[key]]
    elif rev in FEATURE_MAPPING:
        rows = [(t, s, mt, source_name, target_name)
                for s, t, mt in FEATURE_MAPPING[rev]]
    else:
        raise ValueError(f"No mapping for {source_name} -> {target_name}")

    # Mark which pairs were actually active in the fitted harmonizer
    active_pairs: set = set()
    if harmonizer is not None and harmonizer.mapped_pairs:
        active_pairs = {(s, t) for s, t, _ in harmonizer.mapped_pairs}

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        import csv as _csv
        w = _csv.DictWriter(
            f,
            fieldnames=["source_feature", "target_feature", "mapping_type",
                        "source_dataset", "target_dataset", "active_in_data"],
        )
        w.writeheader()
        for src_f, tgt_f, mtype, src_ds, tgt_ds in rows:
            w.writerow({
                "source_feature":  src_f,
                "target_feature":  tgt_f,
                "mapping_type":    mtype,
                "source_dataset":  src_ds,
                "target_dataset":  tgt_ds,
                "active_in_data":  (src_f, tgt_f) in active_pairs
                                   if active_pairs else "unknown",
            })

    print(f"Mapping table saved -> {out_path}  ({len(rows)} rows)")
    return out_path


# ── Visualize feature distributions ───────────────────────────────────────────

def visualize_feature_distributions(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    harmonizer: FeatureHarmonizer,
    top_n: int = 5,
) -> Path:
    """
    Side-by-side histograms comparing the top ``top_n`` semantically mapped
    feature distributions: source (before harmonization), target (reference),
    and harmonized source (after transform).

    Saved to ``results/figures/feature_distributions.png``.
    """
    if not harmonizer.mapped_pairs:
        raise RuntimeError("Call harmonizer.fit() before visualizing.")

    # Pick top_n semantic/identical pairs
    semantic = [(s, t, mt) for s, t, mt in harmonizer.mapped_pairs
                if mt in ("identical", "semantic")][:top_n]
    if not semantic:
        semantic = harmonizer.mapped_pairs[:top_n]

    # Build a small source tensor for transform
    src_cols = harmonizer.source_cols
    numeric_src = source_df[src_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    arr_src = numeric_src.values.astype(np.float32)
    mn, mx = arr_src.min(0, keepdims=True), arr_src.max(0, keepdims=True)
    denom = np.where((mx - mn) > 0, mx - mn, 1.0)
    arr_src_norm = (arr_src - mn) / denom

    tgt_cols = harmonizer.target_cols
    numeric_tgt = target_df[tgt_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    arr_tgt = numeric_tgt.values.astype(np.float32)
    mn_t, mx_t = arr_tgt.min(0, keepdims=True), arr_tgt.max(0, keepdims=True)
    denom_t = np.where((mx_t - mn_t) > 0, mx_t - mn_t, 1.0)
    arr_tgt_norm = (arr_tgt - mn_t) / denom_t

    src_tensor = torch.tensor(arr_src_norm, dtype=torch.float32).to(harmonizer.device)
    harmonized = harmonizer.transform(src_tensor).cpu().numpy()

    n_pairs = len(semantic)
    fig, axes = plt.subplots(n_pairs, 3, figsize=(13, 3 * n_pairs))
    if n_pairs == 1:
        axes = axes[np.newaxis, :]

    cols = ["Source (before)", "Target (reference)", "Harmonized (after)"]
    for col_idx, title in enumerate(cols):
        axes[0, col_idx].set_title(title, fontsize=11, fontweight="bold")

    for row, (s_feat, t_feat, mtype) in enumerate(semantic):
        s_fi = src_cols.index(s_feat)
        t_fi = tgt_cols.index(t_feat)

        src_vals  = arr_src_norm[:, s_fi]
        tgt_vals  = arr_tgt_norm[:, t_fi]
        harm_vals = harmonized[:, t_fi]

        for col_idx, (vals, color) in enumerate(
            [(src_vals, "#4C72B0"), (tgt_vals, "#55A868"), (harm_vals, "#C44E52")]
        ):
            ax = axes[row, col_idx]
            ax.hist(vals, bins=40, color=color, alpha=0.75, edgecolor="none")
            ax.set_ylabel(f"{s_feat}\n-> {t_feat}", fontsize=8)
            ax.set_xlabel("normalised value", fontsize=8)
            ax.tick_params(labelsize=7)
            label = f"[{mtype}]"
            ax.text(0.97, 0.95, label, transform=ax.transAxes,
                    ha="right", va="top", fontsize=7, color="grey")

    fig.suptitle(
        f"Feature distributions: {harmonizer.source_name} -> {harmonizer.target_name}",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()

    RESULTS_FIGURES.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_FIGURES / "feature_distributions.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Distribution plot saved -> {out_path}")
    return out_path


# ── __main__ ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    device_str = "CUDA" if torch.cuda.is_available() else "CPU"
    print(f"Device: {device_str}\n")

    # Build synthetic DataFrames that mimic the real column layout
    rng = np.random.default_rng(42)
    N = 2000

    def _synth_df(features: List[str], label_col: str, n: int = N) -> pd.DataFrame:
        df = pd.DataFrame(rng.random((n, len(features))), columns=features)
        df[label_col] = rng.choice(["Normal", "Attack"], n)
        return df

    nsl_df  = _synth_df(NSL_KDD_FEATURES,    "label")
    unsw_df = _synth_df(UNSW_NB15_FEATURES,  "attack_cat")
    cic_df  = _synth_df(CIC_IDS2017_FEATURES,"Label")

    print("=" * 60)
    print("Harmonizer: NSL-KDD -> UNSW-NB15")
    print("=" * 60)
    h1 = FeatureHarmonizer("nsl_kdd", "unsw_nb15")
    h1.fit(nsl_df, unsw_df, projector_epochs=300)

    s = h1.summary()
    print(f"  Source features        : {s['n_source_feat']}")
    print(f"  Target features        : {s['n_target_feat']}")
    print(f"  Semantic mappings      : {s['n_semantic']}")
    print(f"  Projected mappings     : {s['n_projected_map']}")
    print(f"  FC-projected (unmapped): {s['n_fc_projected']}")
    print(f"  Total target covered   : {s['n_total_covered']}")

    src_t = torch.tensor(
        rng.random((32, len(NSL_KDD_FEATURES))).astype(np.float32)
    ).to(h1.device)
    out_t = h1.transform(src_t)
    print(f"\n  transform({tuple(src_t.shape)}) -> {tuple(out_t.shape)}")
    print(f"  Output range: [{out_t.min():.3f}, {out_t.max():.3f}]  (should be [0,1])")

    # Save mapping table
    save_mapping_table("nsl_kdd", "unsw_nb15", harmonizer=h1)

    # Visualize distributions
    print("\nGenerating distribution plot ...")
    visualize_feature_distributions(nsl_df, unsw_df, h1, top_n=5)

    print()
    print("=" * 60)
    print("Harmonizer: NSL-KDD -> CIC-IDS2017")
    print("=" * 60)
    h2 = FeatureHarmonizer("nsl_kdd", "cic_ids2017")
    h2.fit(nsl_df, cic_df, projector_epochs=300)
    s2 = h2.summary()
    print(f"  Semantic mappings      : {s2['n_semantic']}")
    print(f"  Projected mappings     : {s2['n_projected_map']}")
    print(f"  FC-projected (unmapped): {s2['n_fc_projected']}")
    print(f"  Total target covered   : {s2['n_total_covered']}")

    print()
    print("=" * 60)
    print("Harmonizer: UNSW-NB15 -> CIC-IDS2017")
    print("=" * 60)
    h3 = FeatureHarmonizer("unsw_nb15", "cic_ids2017")
    h3.fit(unsw_df, cic_df, projector_epochs=300)
    s3 = h3.summary()
    print(f"  Semantic mappings      : {s3['n_semantic']}")
    print(f"  Projected mappings     : {s3['n_projected_map']}")
    print(f"  FC-projected (unmapped): {s3['n_fc_projected']}")
    print(f"  Total target covered   : {s3['n_total_covered']}")

    print("\nDone.")
