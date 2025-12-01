BMO Credit Risk Modeling Case Study — PD Model Development

This project simulates a real credit-risk modeling workflow using the BMO ML CoE (Machine Learning Centre of Excellence) case study dataset. It recreates the typical steps used in developing Probability of Default (PD) models in banking: data cleaning, feature engineering, model development, validation, and performance comparison.

🔍 Objective

Predict customer default risk and evaluate the performance of several PD model candidates.

📊 1. Data Preparation & Cleaning

Loaded the official BMO case-study dataset

Inspected data quality using .info(), distribution checks, and missing-value analysis

Cleaned abnormal values and inconsistent fields

Normalized column names and removed duplicated entries

🔧 2. Feature Engineering

Handled categorical variables (encoding/cleaning where needed)

Scaled numerical features using MinMaxScaler / StandardScaler

Conducted correlation analysis to understand key risk drivers

🤖 3. Model Development

Models Developed:

Logistic Regression (baseline PD model)

Random Forest

Gradient Boosting Classifier

Workflows:

Train/test split

Model fitting

Predicting probabilities (predict_proba)

Hyperparameter adjustments for stability

📈 4. Model Evaluation

Metrics used:

ROC-AUC

Precision-Recall Curve

Confusion Matrix

Probability Calibration

Model comparison charts

This mirrors standard PD validation frameworks in banking.

📝 5. Key Insights

Logistic Regression provided a stable, interpretable baseline

Ensemble models boosted predictive power but introduced complexity trade-offs

Feature scaling significantly improved model stability

Identified top predictive drivers consistent with real credit-risk factors


