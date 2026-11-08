# Week 7 — Deployment threshold decision

## Business preference

The stated operating preference is to prioritize recall for churners: missing a customer who is about to leave is considered more costly than contacting some customers who would have stayed. The validation selection rule first seeks recall of at least 75%, then chooses the threshold with the highest precision among thresholds meeting that recall floor. If no threshold meets the floor, it selects the maximum-F2 threshold (F2 weights recall more heavily).

## Threshold selected without holdout leakage

- Selected imbalance strategy: **scale_pos_weight**
- Selected threshold: **0.3251** (different from default 0.50: **True**)
- Selection basis: Highest validation precision among thresholds with recall >= 75%.

Threshold selection used a validation split carved from the training partition. The final stratified holdout remained untouched until final evaluation.

## Final untouched-holdout results

| Metric | Selected threshold | Default 0.50 |
|---|---:|---:|
| precision | 0.5512 | 0.6648 |
| recall | 0.7259 | 0.5054 |
| f1 | 0.6266 | 0.5742 |
| roc_auc | 0.8466 | 0.8466 |
| pr_auc_average_precision | 0.6479 | 0.6479 |
| accuracy | 0.7706 | 0.8012 |

Confusion matrix order is TN, FP, FN, TP.

- At selected threshold: `[1018, 276, 128, 339]`
- At 0.50: `[1175, 119, 231, 236]`

## Strategy comparison

The two imbalance strategies were compared on the same untouched holdout at threshold 0.50. The selected strategy was chosen using validation PR-AUC, not holdout metrics.

| Strategy | Precision | Recall | F1 | ROC-AUC | PR-AUC (AP) |
|---|---:|---:|---:|---:|---:|
| scale_pos_weight | 0.6648 | 0.5054 | 0.5742 | 0.8466 | 0.6479 |
| random_oversampling | 0.5302 | 0.7901 | 0.6346 | 0.8451 | 0.6449 |

The class-weight strategy had the stronger validation PR-AUC and was selected for the threshold-tuned final model. Random oversampling produced higher holdout recall at 0.50 in this run, but also more false positives; the strategy choice was made on validation data to avoid selecting based on holdout outcomes. Full tuned parameters are in `outputs/model_metrics.json`.

## Deployment caveat

This threshold is a model operating point, not a universal optimum. Before deployment, confirm the relative costs and capacity of retention outreach, monitor precision/recall and calibration over time, and revisit the threshold when churn prevalence or intervention costs change.
