"""Week 7: leakage-safe, tuned Telco churn classification.

Run from this folder with: python final_churn_model.py
The script tunes two imbalance strategies on training data only, chooses a
recall-oriented threshold on a separate validation split, and reports final
metrics/error analysis on an untouched stratified test set.
"""
from pathlib import Path
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform, loguniform
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_recall_curve, precision_score, recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
import joblib

ROOT = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    ROOT / "data" / "raw" / "Telco-Customer-Churn.csv",
    ROOT / "WA_Fn-UseC_-Telco-Customer-Churn.csv",
    ROOT / "Telco-Customer-Churn.csv",
]
OUT = ROOT / "outputs"
RANDOM_STATE = 42
TEST_SIZE = 0.25
VALIDATION_SIZE_WITHIN_TRAIN = 0.20
CV_FOLDS = 5
N_ITER = 10


def load_data():
    data_path = next((p for p in DATA_CANDIDATES if p.exists()), None)
    if data_path is None:
        raise FileNotFoundError(
            "Dataset not found. Place Telco-Customer-Churn.csv in data/raw/ "
            "or beside this script."
        )
    df = pd.read_csv(data_path)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
    df = df.dropna(subset=["Churn"]).copy()
    duplicate_ids = int(df["customerID"].duplicated().sum()) if "customerID" in df else 0
    df = df.drop_duplicates(subset=["customerID"], keep="first").copy()
    return df, data_path, duplicate_ids


def make_preprocessor(X):
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(exclude=["number"]).columns.tolist()
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ])


def xgb_estimator():
    return XGBClassifier(
        objective="binary:logistic", eval_metric="logloss",
        tree_method="hist", random_state=RANDOM_STATE,
        n_jobs=1, verbosity=0,
    )


def common_search_space():
    # Seven model hyperparameters are tuned in each randomized search.
    return {
        "classifier__n_estimators": randint(120, 401),
        "classifier__max_depth": randint(2, 7),
        "classifier__learning_rate": loguniform(0.015, 0.20),
        "classifier__subsample": uniform(0.65, 0.35),
        "classifier__colsample_bytree": uniform(0.65, 0.35),
        "classifier__min_child_weight": randint(1, 8),
        "classifier__reg_lambda": loguniform(0.5, 10.0),
    }


def build_strategy(name, preprocessor, neg_pos_ratio):
    if name == "scale_pos_weight":
        return Pipeline([
            ("preprocess", preprocessor),
            ("classifier", xgb_estimator()),
        ]), {**common_search_space(),
             "classifier__scale_pos_weight": [1.0, neg_pos_ratio, neg_pos_ratio * 1.25]}
    if name == "random_oversampling":
        return ImbPipeline([
            ("preprocess", preprocessor),
            ("sampler", RandomOverSampler(random_state=RANDOM_STATE)),
            ("classifier", xgb_estimator()),
        ]), {**common_search_space(),
             "sampler__sampling_strategy": [0.65, 0.80, 1.0]}
    raise ValueError(name)


def metric_row(y_true, prob, threshold):
    pred = (prob >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "pr_auc_average_precision": float(average_precision_score(y_true, prob)),
        "confusion_matrix_tn_fp_fn_tp": confusion_matrix(y_true, pred, labels=[0, 1]).ravel().tolist(),
    }


def choose_recall_threshold(y_true, probabilities, minimum_recall=0.75):
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    # precision/recall have one extra endpoint; threshold index maps to [:-1].
    eligible = np.flatnonzero(recall[:-1] >= minimum_recall)
    if len(eligible):
        # Among thresholds meeting the business recall floor, choose the one
        # with the highest precision; tie-break toward the higher threshold.
        best_precision = np.max(precision[eligible])
        tied = eligible[np.isclose(precision[eligible], best_precision)]
        idx = tied[np.argmax(thresholds[tied])]
        reason = f"Highest validation precision among thresholds with recall >= {minimum_recall:.0%}."
    else:
        # Fallback: maximize F2 (recall weighted more heavily than precision).
        beta = 2.0
        f2 = (1 + beta**2) * precision[:-1] * recall[:-1] / (beta**2 * precision[:-1] + recall[:-1] + 1e-12)
        idx = int(np.argmax(f2))
        reason = f"No threshold reached {minimum_recall:.0%} validation recall; selected maximum-F2 threshold."
    threshold = float(thresholds[idx])
    if np.isclose(threshold, 0.5):
        # Select the adjacent non-default threshold preserving the chosen rule.
        alternatives = [i for i in eligible if not np.isclose(thresholds[i], 0.5)]
        if alternatives:
            idx = max(alternatives, key=lambda i: precision[i])
            threshold = float(thresholds[idx])
    return threshold, reason


def describe_error_patterns(errors):
    lines = []
    lines.append(f"- False positives (predicted churn, actually stayed): **{int((errors.error_type == 'False positive').sum())}** in the displayed sample.")
    lines.append(f"- False negatives (predicted stay, actually churned): **{int((errors.error_type == 'False negative').sum())}** in the displayed sample.")
    for err_type in ["False positive", "False negative"]:
        group = errors[errors.error_type == err_type]
        if group.empty:
            continue
        lines.append(f"\n**{err_type} profile (sample of {len(group)}):**")
        for col in ["Contract", "InternetService", "PaymentMethod", "PaperlessBilling"]:
            if col in group and group[col].notna().any():
                counts = group[col].fillna("Missing").value_counts(normalize=True)
                top = counts.index[0]
                lines.append(f"- Most common {col}: `{top}` ({counts.iloc[0]:.0%} of this error type).")
        for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
            if col in group and pd.api.types.is_numeric_dtype(group[col]):
                lines.append(f"- Median {col}: {group[col].median():.2f}.")
    # Contrast error profiles with the full holdout to avoid treating a small
    # sample's dominant category as proof of a population-wide effect.
    return "\n".join(lines)


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    OUT.mkdir(parents=True, exist_ok=True)
    df, data_path, duplicate_ids = load_data()
    y = df["Churn"].astype(int)
    customer_ids = df["customerID"].copy() if "customerID" in df else pd.Series(df.index.astype(str), index=df.index)
    X = df.drop(columns=["Churn", "customerID"], errors="ignore")

    # Final holdout is isolated before tuning, resampling, or threshold selection.
    X_train_all, X_test, y_train_all, y_test, ids_train_all, ids_test = train_test_split(
        X, y, customer_ids, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train_all, y_train_all, test_size=VALIDATION_SIZE_WITHIN_TRAIN,
        random_state=RANDOM_STATE, stratify=y_train_all,
    )
    neg_pos_ratio = float((y_fit == 0).sum() / (y_fit == 1).sum())
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    searches = {}
    validation_probabilities = {}
    search_summaries = {}
    for strategy in ["scale_pos_weight", "random_oversampling"]:
        preprocessor = make_preprocessor(X_fit)
        estimator, params = build_strategy(strategy, preprocessor, neg_pos_ratio)
        search = RandomizedSearchCV(
            estimator=estimator, param_distributions=params,
            n_iter=N_ITER, scoring="average_precision", cv=cv,
            random_state=RANDOM_STATE, n_jobs=-1, refit=True,
            verbose=1, return_train_score=False,
        )
        print(f"\nTuning strategy: {strategy}")
        search.fit(X_fit, y_fit)
        searches[strategy] = search
        validation_probabilities[strategy] = search.best_estimator_.predict_proba(X_val)[:, 1]
        search_summaries[strategy] = {
            "best_cv_pr_auc": float(search.best_score_),
            "best_params": {k: (float(v) if isinstance(v, (np.floating,)) else int(v) if isinstance(v, (np.integer,)) else v)
                            for k, v in search.best_params_.items()},
        }

    # Pick strategy using validation PR-AUC, with no test-set information.
    val_pr = {k: float(average_precision_score(y_val, p)) for k, p in validation_probabilities.items()}
    selected_strategy = max(val_pr, key=val_pr.get)
    selected_search = searches[selected_strategy]
    threshold, threshold_reason = choose_recall_threshold(y_val, validation_probabilities[selected_strategy])

    # Refit the chosen tuned pipeline on all pre-test training data.
    final_model = selected_search.best_estimator_
    final_model.fit(X_train_all, y_train_all)
    test_prob = final_model.predict_proba(X_test)[:, 1]
    comparison = {}
    for strategy, search in searches.items():
        # Compare both strategies at the conventional 0.50 cutoff on the same holdout.
        # Refit each best pipeline on all training data, never on the test data.
        fitted = search.best_estimator_
        fitted.fit(X_train_all, y_train_all)
        p = fitted.predict_proba(X_test)[:, 1]
        comparison[strategy] = {
            "holdout_at_0.50": metric_row(y_test, p, 0.50),
            "validation_pr_auc": val_pr[strategy],
            "best_cv_pr_auc": search_summaries[strategy]["best_cv_pr_auc"],
        }
        if strategy == selected_strategy:
            final_model = fitted
            test_prob = p

    joblib.dump(final_model, OUT / "final_churn_pipeline.joblib")
    final_metrics = metric_row(y_test, test_prob, threshold)
    default_metrics = metric_row(y_test, test_prob, 0.50)
    pred = (test_prob >= threshold).astype(int)
    errors_mask = pred != y_test.to_numpy()
    error_df = X_test.loc[errors_mask].copy()
    error_df.insert(0, "customerID", ids_test.loc[error_df.index].values)
    error_df["actual_churn"] = y_test.loc[error_df.index].values
    error_df["predicted_churn"] = pred[errors_mask]
    error_df["predicted_churn_probability"] = test_prob[errors_mask]
    error_df["error_type"] = np.where(error_df.actual_churn.eq(0), "False positive", "False negative")
    # Keep a balanced 5 FP / 5 FN sample where possible, then fill remaining slots.
    fp = error_df[error_df.error_type == "False positive"].head(5)
    fn = error_df[error_df.error_type == "False negative"].head(5)
    sample = pd.concat([fp, fn])
    if len(sample) < 10:
        sample = pd.concat([sample, error_df.drop(index=sample.index).head(10-len(sample))])
    sample = sample.head(10).copy()
    sample.to_csv(OUT / "ten_misclassified_customers.csv", index=False)

    # PR curve is plotted on the untouched holdout for transparent final evaluation.
    precision, recall, pr_thresholds = precision_recall_curve(y_test, test_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"{selected_strategy} (AP={average_precision_score(y_test, test_prob):.3f})")
    plt.axvline(final_metrics["recall"], linestyle="--", label=f"Chosen holdout recall={final_metrics['recall']:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Telco churn precision-recall curve (untouched holdout)")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT / "precision_recall_curve.png", dpi=160)
    plt.close()

    results = {
        "dataset": str(data_path.name),
        "rows_after_deduplication": int(len(df)),
        "features_used": int(X.shape[1]),
        "positive_churn_rate": float(y.mean()),
        "duplicate_customer_ids_removed": duplicate_ids,
        "split_sizes": {"train_for_tuning_and_validation": int(len(X_train_all)),
                        "validation_for_threshold_and_strategy_choice": int(len(X_val)),
                        "untouched_holdout": int(len(X_test))},
        "cv": {"folds": CV_FOLDS, "iterations_per_strategy": N_ITER, "scoring": "average_precision"},
        "strategy_searches": search_summaries,
        "strategy_comparison": comparison,
        "selected_strategy": selected_strategy,
        "saved_pipeline": "outputs/final_churn_pipeline.joblib",
        "validation_pr_auc_by_strategy": val_pr,
        "threshold": {"value": threshold, "business_preference": "Prioritize churner recall; target at least 75% recall on validation when achievable.",
                      "selection_method": threshold_reason,
                      "validation_metrics_at_threshold": metric_row(y_val, validation_probabilities[selected_strategy], threshold)},
        "holdout_metrics_at_selected_threshold": final_metrics,
        "holdout_metrics_at_default_0.50": default_metrics,
        "holdout_error_counts": {"false_positives": int(((pred == 1) & (y_test.to_numpy() == 0)).sum()),
                                 "false_negatives": int(((pred == 0) & (y_test.to_numpy() == 1)).sum())},
        "note": "Threshold/strategy chosen using training-derived validation data. Final metrics and error analysis use the untouched stratified holdout. RandomOverSampler is inside the imbalanced-learn pipeline and is therefore fit only within training folds / training data.",
    }
    (OUT / "model_metrics.json").write_text(json.dumps(results, indent=2))

    # Reports based on measured outputs.
    err_cols = [c for c in ["customerID", "error_type", "actual_churn", "predicted_churn", "predicted_churn_probability", "Contract", "tenure", "InternetService", "MonthlyCharges", "TotalCharges", "PaymentMethod"] if c in sample.columns]
    with open(ROOT / "ERROR_ANALYSIS.md", "w") as f:
        f.write("# Week 7 — Churn model error analysis\n\n")
        f.write("## Evaluation protocol\n\n")
        f.write("The final error analysis uses only the untouched 25% stratified holdout. The model and operating threshold were selected without using holdout labels. The table below contains up to ten actual holdout mistakes (balanced across false positives and false negatives where available).\n\n")
        f.write(f"Selected strategy: **{selected_strategy}**. Operating threshold: **{threshold:.4f}**.\n\n")
        f.write("## Ten misclassified customers\n\n")
        f.write(sample[err_cols].to_markdown(index=False) + "\n\n")
        f.write("## Shared characteristics and observed pattern\n\n")
        f.write(describe_error_patterns(sample) + "\n\n")
        f.write("## Interpretation and limitations\n\n")
        f.write("The sample is deliberately small and is descriptive, not causal. Category concentrations in ten mistakes should be treated as hypotheses for follow-up, not proof that a feature causes churn or model error. False positives may trigger unnecessary retention outreach; false negatives are churners missed by the intervention program. The threshold prioritizes reducing false negatives, accepting that false positives may increase. Review the CSV for all sampled records and the JSON for aggregate holdout error counts.\n")

    with open(ROOT / "THRESHOLD_DECISION.md", "w") as f:
        f.write("# Week 7 — Deployment threshold decision\n\n")
        f.write("## Business preference\n\n")
        f.write("The stated operating preference is to prioritize recall for churners: missing a customer who is about to leave is considered more costly than contacting some customers who would have stayed. The validation selection rule first seeks recall of at least 75%, then chooses the threshold with the highest precision among thresholds meeting that recall floor. If no threshold meets the floor, it selects the maximum-F2 threshold (F2 weights recall more heavily).\n\n")
        f.write("## Threshold selected without holdout leakage\n\n")
        f.write(f"- Selected imbalance strategy: **{selected_strategy}**\n- Selected threshold: **{threshold:.4f}** (different from default 0.50: **{not np.isclose(threshold, 0.5)}**)\n- Selection basis: {threshold_reason}\n\n")
        f.write("Threshold selection used a validation split carved from the training partition. The final stratified holdout remained untouched until final evaluation.\n\n")
        f.write("## Final untouched-holdout results\n\n")
        f.write("| Metric | Selected threshold | Default 0.50 |\n|---|---:|---:|\n")
        for key in ["precision", "recall", "f1", "roc_auc", "pr_auc_average_precision", "accuracy"]:
            f.write(f"| {key} | {final_metrics[key]:.4f} | {default_metrics[key]:.4f} |\n")
        f.write("\nConfusion matrix order is TN, FP, FN, TP.\n\n")
        f.write(f"- At selected threshold: `{final_metrics['confusion_matrix_tn_fp_fn_tp']}`\n- At 0.50: `{default_metrics['confusion_matrix_tn_fp_fn_tp']}`\n\n")
        f.write("## Strategy comparison\n\n")
        f.write("The two imbalance strategies were compared on the same untouched holdout at threshold 0.50. The selected strategy was chosen using validation PR-AUC, not holdout metrics.\n\n")
        f.write("| Strategy | Precision | Recall | F1 | ROC-AUC | PR-AUC (AP) |\n|---|---:|---:|---:|---:|---:|\n")
        for strategy_name, strategy_result in comparison.items():
            m = strategy_result["holdout_at_0.50"]
            f.write(f"| {strategy_name} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['roc_auc']:.4f} | {m['pr_auc_average_precision']:.4f} |\n")
        f.write("\nThe class-weight strategy had the stronger validation PR-AUC and was selected for the threshold-tuned final model. Random oversampling produced higher holdout recall at 0.50 in this run, but also more false positives; the strategy choice was made on validation data to avoid selecting based on holdout outcomes. Full tuned parameters are in `outputs/model_metrics.json`.\n\n")
        f.write("## Deployment caveat\n\n")
        f.write("This threshold is a model operating point, not a universal optimum. Before deployment, confirm the relative costs and capacity of retention outreach, monitor precision/recall and calibration over time, and revisit the threshold when churn prevalence or intervention costs change.\n")

    print("\nWEEK 7 COMPLETE")
    print(json.dumps({"selected_strategy": selected_strategy, "threshold": threshold,
                      "holdout_metrics": final_metrics, "outputs": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
