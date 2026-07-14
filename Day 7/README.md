# Day 7 – Random Forest Classifier

## Project Objective

The objective of this project is to build a Random Forest Classifier using Scikit-Learn and compare its performance with the baseline Logistic Regression model created previously.

---

## Dataset

The project uses the cleaned and One-Hot Encoded Titanic dataset created during Week 1.

Target Variable:

- Survived

---

## Features (X)

The model uses the following independent variables:

- Pclass
- Age
- SibSp
- Parch
- Fare
- Sex_Male
- Embarked_Q
- Embarked_S

---

## Target (y)

- Survived

0 = Did Not Survive

1 = Survived

---

## Model

Random Forest Classifier

Parameters used:

- n_estimators = 100
- random_state = 42

---

## Evaluation Metrics

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

Replace the values below with your actual results:

- Accuracy:
- Precision:
- Recall:
- F1-Score:

---

## Logistic Regression vs Random Forest

Random Forest combines multiple Decision Trees to improve prediction performance and reduce overfitting. Compared to the baseline Logistic Regression model, it can capture more complex and non-linear relationships in the data.

---

## Libraries Used

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib
- jupyter

---

## How to Run

1. Install the required libraries.
2. Open `random_forest_model.ipynb`.
3. Run all cells from top to bottom.
4. The notebook will train the model, evaluate it, and save `random_forest_model.pkl`.

---

## Author

Ahmed Ali

BS Artificial Intelligence

ProSensia AI/ML Internship