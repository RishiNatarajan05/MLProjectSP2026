# FAA Wildlife Strike Damage — Binary Classification

Predict `INDICATED_DAMAGE` (1 = damage, 0 = no damage) from FAA wildlife-strike incident records. Scoring metric is **Balanced Accuracy** (mean of per-class recall).

The pipeline uses temporal validation (`INCIDENT_YEAR == 2015` held out), Optuna hyper-parameter tuning, threshold sweeping for Balanced Accuracy, and a stacked ensemble of LightGBM / CatBoost / XGBoost / TF-IDF logistic regression.

## Repository contents

| File | Purpose |
|---|---|
| `build_notebook.py` | Generator script that builds `damage_prediction.ipynb` programmatically via `nbformat`. |
| `damage_prediction.ipynb` | Full executed notebook with EDA, cleaning, feature engineering, modeling, interpretability, and submission. |
| `submission.csv` | Final predictions for the test set, ready to upload. |
| `submission_v2_catboost.csv` | Backup submission from a non-TF-IDF run. |
| `requirements.txt` | Pinned Python dependencies. |
| `.gitignore` | Excludes `.venv/`, large CSVs, and execution logs. |

`train.csv`, `test.csv`, and `sample_submission.csv` are **not** committed (each is over GitHub's file-size limit). Download them from the competition page and drop them in the repo root before running.

## Setup

Requires Python **3.11**.

```bash
git clone https://github.com/RishiNatarajan05/MLProjectSP2026.git
cd MLProjectSP2026

# create venv and install dependencies
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Place the competition data in the repo root:

```
./train.csv
./test.csv
./sample_submission.csv
```

## Running the pipeline

End-to-end (regenerate notebook from `build_notebook.py`, then execute it):

```bash
python build_notebook.py
jupyter nbconvert --to notebook \
                  --execute damage_prediction.ipynb \
                  --output damage_prediction.ipynb \
                  --ExecutePreprocessor.timeout=3600
```

This produces `submission.csv` in the working directory. Total runtime is roughly **20–30 minutes** on a laptop CPU.

If you prefer interactive use, open the notebook directly:

```bash
jupyter lab damage_prediction.ipynb
```

and run **Cell → Run All**.

## What the notebook does

1. **Setup & Load** — imports, seeding (`SEED = 42`), data loading.
2. **EDA** — target distribution, missingness, damage rate by phase / species / airport / height / etc.
3. **Cleaning** — century-aware date parsing, Excel-corrupted `NUM_STRUCK` reverse-mapping, multi-hot precipitation, 99.5th-percentile clipping with `log1p` copies, median imputation within `PHASE_OF_FLIGHT`, missingness indicators, `TIME` → hour.
4. **Feature engineering** — frequency + 5-fold smoothed target-mean encodings for high-cardinality categoricals, cyclical month / hour, day of week, mass × size and phase × height interactions, engine count, species / airport historical damage rate, hand-crafted text features and TF-IDF over `REMARKS + COMMENTS`.
5. **Modeling** — balanced logistic baseline, LightGBM, CatBoost (native categorical handling), XGBoost, TF-IDF logistic regression, MLP with `alpha` tuning, plus an honest "clean-features-only" LightGBM. Each model is tuned with Optuna (≥ 30 trials) and has its decision threshold swept on the 2015 fold.
6. **Model comparison table** — Balanced Accuracy at the default and tuned thresholds, ROC-AUC, PR-AUC.
7. **Interpretability** — top-30 LightGBM importances and a SHAP summary plot on a 5 000-row validation sample.
8. **Final fit & submission** — refit the winning model on all 1990–2015 training rows with frozen hyper-parameters, apply the threshold picked on 2015, write `submission.csv`.

## Reproducibility

All random seeds are pinned to `42`. Optuna uses `TPESampler(seed=42)`. Boosters are seeded identically. Pandas / sklearn versions are pinned in `requirements.txt`. Re-running the pipeline on the same data should yield byte-identical predictions.

## Caveats

A handful of features (`REMARKS`, `COMMENTS`, `REMAINS_*`, `BIRD_BAND_NUMBER`) are populated post-event by investigators and effectively leak the label. They are present in `test.csv` and competition-legal, so the main model uses them. The "clean-features-only" LightGBM in the comparison table quantifies the gap if you wanted an operational pre-event model instead.
