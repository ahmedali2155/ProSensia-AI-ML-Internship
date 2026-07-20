# Loan Default Prediction

## Objective
Build an end-to-end machine learning classification pipeline for loan default prediction.

## Dataset
Analytics Vidhya Loan Prediction Dataset

## Steps Performed
- Data Cleaning
- Missing Value Handling
- Label Encoding
- Train-Test Split
- SMOTE
- Logistic Regression
- Decision Tree
- Random Forest
- ROC-AUC Comparison
- Best Model Selection

## Model Performance

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|---------|----------|-------|------|---------|
| Logistic Regression | 0.8130 | 0.8605 | 0.8706 | **0.8655** | **0.8514** |
| Decision Tree | 0.6992 | 0.8333 | 0.7059 | 0.7643 | 0.6950 |
| Random Forest | 0.8049 | 0.8506 | 0.8706 | 0.8605 | 0.7915 |

## Best Model
Logistic Regression was selected because it achieved the highest ROC-AUC score while maintaining the best balance of Precision, Recall, and F1-score.

## Libraries
pandas, numpy, matplotlib, seaborn, scikit-learn, imbalanced-learn, joblib

## Author
Ahmed Ali