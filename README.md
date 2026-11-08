# Week 7 — Tuned, Pipelined Churn Classification

## Deliverables
- `final_churn_model.py` — complete end-to-end training, tuning, threshold selection, evaluation and error-analysis script.
- `ERROR_ANALYSIS.md` — structured analysis of ten holdout mistakes.
- `THRESHOLD_DECISION.md` — recall-oriented threshold decision and measured holdout results.
- `outputs/final_churn_pipeline.joblib` — fitted preprocessing + XGBoost pipeline.
- `outputs/model_metrics.json` — search parameters, strategy comparison and final metrics.
- `outputs/precision_recall_curve.png` — precision-recall curve on the untouched holdout.
- `outputs/ten_misclassified_customers.csv` — the ten sampled holdout mistakes.

## Run
Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
python final_churn_model.py
```

The script expects the dataset at `data/raw/Telco-Customer-Churn.csv` (included in this project). It uses a stratified 25% untouched test set, splits the remaining training data to create a validation set for strategy/threshold selection, and performs 5-fold `RandomizedSearchCV` on the fit portion. Random oversampling is placed inside an imbalanced-learn pipeline so it is applied only to training folds / training data.

## Reproducibility
Random state is fixed at 42. The randomized search uses 10 candidates per imbalance strategy and tunes seven XGBoost hyperparameters plus the strategy-specific imbalance parameter.
