# Predictive Machine Maintenance Using Machine Learning and Deep Learning Models(A Study on the AI4I 2020 Predictive Maintenance Dataset)

Machine failure prediction using seven machine learning and deep learning models, evaluated on the AI4I 2020 Predictive Maintenance Dataset, with SHAP-based explainability on the best-performing model.

## Overview

Unplanned machine failure is a major source of downtime and cost in industrial manufacturing. This project trains and compares **Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM, CatBoost, and a Multi-Layer Perceptron (MLP)** to predict binary machine failure from operational sensor readings, then uses **SHAP (SHapley Additive exPlanations)** to interpret which features drive the best model's predictions.

**Best model: XGBoost** — 98.90% accuracy, F1-score of 0.8226, ROC-AUC of 0.9639.

## Dataset

[AI4I 2020 Predictive Maintenance Dataset](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification), 10,000 records, 10 columns (Kaggle redistribution of the original UCI dataset, which consolidates five binary failure-mode indicators into two fields: `Target` and `Failure Type`).

Features used for training:
- Air Temperature (K)
- Process Temperature (K)
- Rotational Speed (RPM)
- Torque (Nm)
- Tool Wear (minutes)
- Type (product quality variant: Low / Medium / High)

## Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.9680 | 0.6667 | 0.1176 | 0.2000 | 0.9004 |
| Decision Tree | 0.9795 | 0.6800 | 0.7500 | 0.7133 | 0.8688 |
| Random Forest | 0.9850 | 0.8800 | 0.6471 | 0.7458 | 0.9644 |
| CatBoost | 0.9870 | 0.9038 | 0.6912 | 0.7833 | 0.9723 |
| LightGBM | 0.9875 | 0.8909 | 0.7206 | 0.7967 | 0.9798 |
| **XGBoost** | **0.9890** | **0.9107** | **0.7500** | **0.8226** | 0.9639 |
| MLP (Deep Learning) | 0.9755 | 0.6792 | 0.5294 | 0.5950 | 0.9577 |

*(sorted by F1-score above for readability; see notebook/paper for full ranked table)*

### Explainability (SHAP)

SHAP was applied to the final XGBoost model to identify which features most influence its predictions. **Torque** and **tool wear** are the two strongest drivers of predicted failure, followed by air temperature.

## Repository Structure

```
├── predictive_maintenance.ipynb   # Full pipeline: EDA, preprocessing, 7 models, SHAP
├── REPORT.docx                    # Project report
├── RESEARCH_PAPER.docx            # Research paper writeup
└── README.md
```

## Running the Notebook

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification) or attach it via Kaggle's "Add Input."
2. Update `DATA_PATH` in the first code cell if running outside Kaggle.
3. Run all cells top to bottom. Model comparison charts and SHAP plots are saved automatically as `.png` files.

## Methodology Notes / Limitations

- The `Type` feature is label-encoded as an ordinal integer (0/1/2) rather than one-hot encoded; since the three product variants have no inherent ordering with respect to failure risk, this is a known limitation of the current preprocessing — one-hot encoding or native categorical handling (e.g. CatBoost's built-in support) would be a cleaner alternative.
- The dataset is heavily imbalanced (~96.6% healthy vs. 3.4% failure), so F1-score — not accuracy — was used as the primary model-selection criterion.

## References

1. Bansal, S. *Machine Predictive Maintenance Classification* (Kaggle Dataset).
2. UCI Machine Learning Repository. *AI4I 2020 Predictive Maintenance Dataset*.
3. Waghulde, R. R., et al. (2025). *Evaluating Machine Learning and Deep Learning Models for Predictive Maintenance: A Study Using the AI4I 2020 Dataset.* Journal of Neonatal Surgery, 14(31S), 588–594.
4. Matzka, S. (2020). *Explainable Artificial Intelligence for Predictive Maintenance Applications.* IEEE AI4I 2020, 69–74.
5. Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS 30.
