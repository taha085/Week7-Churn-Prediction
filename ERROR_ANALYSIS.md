# Week 7 — Churn model error analysis

## Evaluation protocol

The final error analysis uses only the untouched 25% stratified holdout. The model and operating threshold were selected without using holdout labels. The table below contains up to ten actual holdout mistakes (balanced across false positives and false negatives where available).

Selected strategy: **scale_pos_weight**. Operating threshold: **0.3251**.

## Ten misclassified customers

| customerID   | error_type     |   actual_churn |   predicted_churn |   predicted_churn_probability | Contract       |   tenure | InternetService   |   MonthlyCharges |   TotalCharges | PaymentMethod             |
|:-------------|:---------------|---------------:|------------------:|------------------------------:|:---------------|---------:|:------------------|-----------------:|---------------:|:--------------------------|
| 1628-BIZYP   | False positive |              0 |                 1 |                     0.860569  | Month-to-month |        1 | Fiber optic       |            85    |          85    | Electronic check          |
| 0537-QYZZN   | False positive |              0 |                 1 |                     0.428129  | Month-to-month |       24 | Fiber optic       |            79.85 |        1857.75 | Electronic check          |
| 8778-LMWTJ   | False positive |              0 |                 1 |                     0.495121  | Month-to-month |        9 | Fiber optic       |            74.85 |         708.2  | Electronic check          |
| 4912-PIGUY   | False positive |              0 |                 1 |                     0.90489   | Month-to-month |        1 | Fiber optic       |            84.6  |          84.6  | Electronic check          |
| 8033-VCZGH   | False positive |              0 |                 1 |                     0.405066  | One year       |       50 | Fiber optic       |           103.95 |        5231.3  | Electronic check          |
| 0461-CVKMU   | False negative |              1 |                 0 |                     0.211222  | Month-to-month |       23 | Fiber optic       |            83.8  |        1900.25 | Electronic check          |
| 9254-RBFON   | False negative |              1 |                 0 |                     0.134712  | One year       |       17 | Fiber optic       |            98.6  |        1704.95 | Bank transfer (automatic) |
| 6227-HWPWX   | False negative |              1 |                 0 |                     0.267439  | Month-to-month |       15 | Fiber optic       |            69    |         994.8  | Bank transfer (automatic) |
| 0733-VUNUW   | False negative |              1 |                 0 |                     0.117212  | Month-to-month |       24 | DSL               |            55.65 |        1400.55 | Bank transfer (automatic) |
| 2091-MJTFX   | False negative |              1 |                 0 |                     0.0913844 | Month-to-month |       30 | DSL               |            51.2  |        1561.5  | Credit card (automatic)   |

## Shared characteristics and observed pattern

- False positives (predicted churn, actually stayed): **5** in the displayed sample.
- False negatives (predicted stay, actually churned): **5** in the displayed sample.

**False positive profile (sample of 5):**
- Most common Contract: `Month-to-month` (80% of this error type).
- Most common InternetService: `Fiber optic` (100% of this error type).
- Most common PaymentMethod: `Electronic check` (100% of this error type).
- Most common PaperlessBilling: `Yes` (100% of this error type).
- Median tenure: 9.00.
- Median MonthlyCharges: 84.60.
- Median TotalCharges: 708.20.

**False negative profile (sample of 5):**
- Most common Contract: `Month-to-month` (80% of this error type).
- Most common InternetService: `Fiber optic` (60% of this error type).
- Most common PaymentMethod: `Bank transfer (automatic)` (60% of this error type).
- Most common PaperlessBilling: `No` (80% of this error type).
- Median tenure: 23.00.
- Median MonthlyCharges: 69.00.
- Median TotalCharges: 1561.50.

## Interpretation and limitations

The sample is deliberately small and is descriptive, not causal. Category concentrations in ten mistakes should be treated as hypotheses for follow-up, not proof that a feature causes churn or model error. False positives may trigger unnecessary retention outreach; false negatives are churners missed by the intervention program. The threshold prioritizes reducing false negatives, accepting that false positives may increase. Review the CSV for all sampled records and the JSON for aggregate holdout error counts.
