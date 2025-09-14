# Marketing Mix Modeling with Mediation Analysis

This repository contains a reproducible implementation for **Assessment 2: MMM Modeling with Mediation Assumption**.

---

## 📂 What’s Included
- Complete, copy-paste-ready Python files to run the analysis.
- A **two-stage (mediator-aware)** modeling approach where **Google spend** is treated as a mediator of social channels → revenue.
- **Time-series-safe cross-validation**, diagnostics, and a short write-up that covers the assessment requirements.

---

## 📁 Files
- `requirements.txt` — list of required Python packages
- `config.py` — constants and global settings
- `data_preprocessing.py` — load and preprocess data (weekly seasonality, zero-spend handling, feature transforms)
- `mediation_model.py` — two-stage mediation model (Google as mediator)
- `model_training.py` — training loop with time-series cross-validation
- `evaluate.py` — residual plots, diagnostics, metrics saving
- `main.py` — orchestrates preprocessing, training, evaluation
- `marketing_data.csv` — your dataset (place in repo root)

---

## ▶️ How to Run
1. **Clone or copy** this repo into a folder (e.g., `mmm-analysis`).
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate       # on Mac/Linux
   venv\Scripts\activate          # on Windows
3.Install dependencies:

pip install -r requirements.txt


4.Place your dataset as marketing_data.csv in the repo root.

5.Run the analysis:

python main.py


6.Outputs (plots, metrics, predictions) are saved in the outputs/ folder.

Data Expectations:

CSV file must include the following columns:

week,facebook_spend,google_spend,tiktok_spend,instagram_spend,snapchat_spend,social_followers,average_price,promotions,emails_send,

Example week format: 17-09-2023 (day-month-year)

Frequency: weekly

Revenue: target variable

Outputs

Running main.py generates:

outputs/cv_results.csv — RMSE & MAE by fold

outputs/predictions_fold_*.csv — per-fold predictions

outputs/predictions_full.csv — full dataset predictions

outputs/residuals_full.png — residual plot

outputs/true_vs_pred_full.png — true vs predicted revenue plot

outputs/metrics_summary.txt — performance summary

