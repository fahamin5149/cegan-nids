# CE-GAN: Cross-Dataset Transferability for Network Intrusion Detection

University project — FAST-NUCES Islamabad

---

## Requirements

- Python 3.11
- CUDA 12.4 compatible GPU (recommended; CPU will work but is very slow)

---

## 1. Clone and Set Up the Environment

```powershell
git clone https://github.com/fahamin5149/cegan-nids.git
cd cegan-nids
```

Create and activate the virtual environment:

```powershell
python -m venv .venv311
```

Activate (run this every time before using the project):

```powershell
.venv311\Scripts\activate
```

Your terminal prompt will change to `(.venv311)` when active.

Install dependencies:

```powershell
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

> If your machine has a different CUDA version, replace `cu124` with your version (e.g. `cu118` for CUDA 11.8). If no GPU, omit the `--index-url` flag entirely.

---

## 2. Dataset Setup

Create the data folders:

```powershell
mkdir data\nsl_kdd
mkdir data\unsw_nb15
mkdir data\cic_ids2017
```

### Dataset 1 — NSL-KDD

**Download from:** https://www.unb.ca/cic/datasets/nsl.html

1. Scroll to the download section and download `NSL-KDD.zip`
2. Extract the zip — you need exactly these two files:
   - `KDDTrain+.txt`
   - `KDDTest+.txt`
3. Place both files in:

```
data/
└── nsl_kdd/
    ├── KDDTrain+.txt
    └── KDDTest+.txt
```

---

### Dataset 2 — UNSW-NB15

**Download from:** https://research.unsw.edu.au/projects/unsw-nb15-dataset

1. Scroll to the "Files" section on the page
2. Download all 4 CSV files:
   - `UNSW-NB15_1.csv`
   - `UNSW-NB15_2.csv`
   - `UNSW-NB15_3.csv`
   - `UNSW-NB15_4.csv`
3. Place all 4 files directly in:

```
data/
└── unsw_nb15/
    ├── UNSW-NB15_1.csv
    ├── UNSW-NB15_2.csv
    ├── UNSW-NB15_3.csv
    └── UNSW-NB15_4.csv
```

> Do NOT rename the files. The loader expects exactly these names.
> Do NOT place them in a subfolder inside `unsw_nb15/`.

---

### Dataset 3 — CIC-IDS2017

**Download via Kaggle.**

First, set up your Kaggle API credentials:

1. Go to https://www.kaggle.com/settings → scroll to "API" → click "Create New Token"
2. This downloads a `kaggle.json` file
3. Place it at `C:\Users\YOUR_USERNAME\.kaggle\kaggle.json`

Then run:

```powershell
.venv311\Scripts\python.exe -m kaggle datasets download mdalamintalukder/cicids2017 --unzip -p data\cic_ids2017
```

This downloads and extracts ~230 MB. When done, your folder should look like:

```
data/
└── cic_ids2017/
    └── MachineLearningCVE/
        ├── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
        ├── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
        ├── Friday-WorkingHours-Morning.pcap_ISCX.csv
        ├── Monday-WorkingHours.pcap_ISCX.csv
        ├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
        ├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
        ├── Tuesday-WorkingHours.pcap_ISCX.csv
        └── Wednesday-workingHours.pcap_ISCX.csv
```

---

## 3. Verify Dataset Setup

Run this to confirm all three datasets load correctly:

```powershell
.venv311\Scripts\python.exe -c "
import sys; sys.path.insert(0, '.')
from src.datasets import DatasetLoader
for name, cap in [('nsl_kdd', None), ('unsw_nb15', 100_000), ('cic_ids2017', 100_000)]:
    dl = DatasetLoader(name, max_samples=cap)
    Xtr, Xte, ytr, yte = dl.load()
    print(f'{name}: train={Xtr.shape}, classes={dl.num_classes}')
"
```

Expected output:
```
nsl_kdd:     train=torch.Size([118800, 41]), classes=34
unsw_nb15:   train=torch.Size([80000,  43]), classes=10
cic_ids2017: train=torch.Size([80000,  78]), classes=12
```

---

## 4. Running the Experiments

The experiment is split across two machines. Both machines need all three datasets set up as above.

The full argument reference:
```
--epochs 100          # GAN training epochs
--batch_size 256
--n_aug 500           # synthetic samples per class
--d_model 128         # transformer hidden size
--n_layers 3          # transformer layers
--n_estimators 100    # trees per classifier in Nash ensemble
--projector_epochs 200
--skip_smote          # skip SMOTE baseline (run separately if needed)
```

---

### Machine 1 (Amin Laptop) — NSL-KDD source scenarios

Activate venv first: `.venv311\Scripts\activate`

Run these **5 scripts in order** (each takes ~1-2 hours):

```powershell
# Scenario A — within-dataset on NSL-KDD
python experiments/run_scenario_a.py --dataset nsl_kdd --epochs 100 --batch_size 256 --n_aug 500 --d_model 128 --n_layers 3 --n_estimators 100

# Scenario B — NSL-KDD as source, transfer to UNSW-NB15
python experiments/run_scenario_b.py --source nsl_kdd --target unsw_nb15 --epochs 100 --batch_size 256 --n_aug 500 --d_model 128 --n_layers 3 --n_estimators 100 --projector_epochs 200

# Scenario B — NSL-KDD as source, transfer to CIC-IDS2017
python experiments/run_scenario_b.py --source nsl_kdd --target cic_ids2017 --epochs 100 --batch_size 256 --n_aug 500 --d_model 128 --n_layers 3 --n_estimators 100 --projector_epochs 200

# Scenario C — NSL-KDD as source, MMD-aligned transfer to UNSW-NB15
python experiments/run_scenario_c.py --source nsl_kdd --target unsw_nb15 --epochs 100 --batch_size 256 --n_aug 500 --d_model 128 --n_layers 3 --n_estimators 100 --projector_epochs 200

# Scenario C — NSL-KDD as source, MMD-aligned transfer to CIC-IDS2017
python experiments/run_scenario_c.py --source nsl_kdd --target cic_ids2017 --epochs 100 --batch_size 256 --n_aug 500 --d_model 128 --n_layers 3 --n_estimators 100 --projector_epochs 200
```

After all 5 finish, commit and push:
```powershell
git add results/tables/
git commit -m "Add Machine 1 results (nsl_kdd source)"
git push
```

---

### Machine 2 (Sarim's Laptop) — UNSW-NB15 source + CIC-IDS2017 Scenario A

Activate venv first: `.venv311\Scripts\activate`

Run these **6 scripts in order**:

```powershell
# Scenario A — within-dataset on UNSW-NB15
python experiments/run_scenario_a.py --dataset unsw_nb15 --epochs 100 --batch_size 256 --n_aug 500 --d_model 128 --n_layers 3 --n_estimators 100

# Scenario A — within-dataset on CIC-IDS2017
python experiments/run_scenario_a.py --dataset cic_ids2017 --epochs 100 --batch_size 256 --n_aug 500 --d_model 128 --n_layers 3 --n_estimators 100

# Scenario B — UNSW-NB15 as source, transfer to NSL-KDD
python experiments/run_scenario_b.py --source unsw_nb15 --target nsl_kdd --epochs 100 --batch_size 256 --n_aug 500 --d_model 128 --n_layers 3 --n_estimators 100 --projector_epochs 200

# Scenario B — UNSW-NB15 as source, transfer to CIC-IDS2017
python experiments/run_scenario_b.py --source unsw_nb15 --target cic_ids2017 --epochs 100 --batch_size 256 --n_aug 500 --d_model 128 --n_layers 3 --n_estimators 100 --projector_epochs 200

# Scenario C — UNSW-NB15 as source, MMD-aligned transfer to NSL-KDD
python experiments/run_scenario_c.py --source unsw_nb15 --target nsl_kdd --epochs 100 --batch_size 256 --n_aug 500 --d_model 128 --n_layers 3 --n_estimators 100 --projector_epochs 200

# Scenario C — UNSW-NB15 as source, MMD-aligned transfer to CIC-IDS2017
python experiments/run_scenario_c.py --source unsw_nb15 --target cic_ids2017 --epochs 100 --batch_size 256 --n_aug 500 --d_model 128 --n_layers 3 --n_estimators 100 --projector_epochs 200
```

After all 6 finish, commit and push:
```powershell
git add results/tables/
git commit -m "Add Machine 2 results (unsw_nb15 source + cic_ids2017 Scenario A)"
git push
```

---

## 5. Merging Results (on Machine 1)

After both machines have pushed their results:

```powershell
git pull
```

The result CSVs from both machines will be merged in `results/tables/table3_main_results.csv`.

---

## Results Location

| File | Contents |
|------|----------|
| `results/tables/table3_main_results.csv` | Per-scenario accuracy, F1, MCC |
| `results/tables/table4_tqs_scores.csv` | Transferability Quality Scores |
| `results/tables/table5_smote_comparison.csv` | SMOTE baseline comparison |
| `results/experiment_log.txt` | Full run log |
