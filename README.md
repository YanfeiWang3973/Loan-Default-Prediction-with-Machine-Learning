
# Loan Default Prediction with Machine Learning

This project explores **predicting loan defaults** using machine learning models.  
The dataset contains borrower information (loan amount, interest rate, annual income, credit history, home ownership, state, etc.) with a binary target variable:  
- `0 = fully paid`  
- `1 = charged off`  

---

Data Preprocessing

Limited dataset to 200,000 rows for optimal performance in Colab’s memory constraints.

Merged multiple CSVs with one-to-many relationships to form a unified dataset.

Cleaned all missing and null values.

Split features into categorical and numerical groups for targeted processing.

Applied one-hot encoding to categorical columns and scaling to numerical ones.

Binned continuous variables for better interpretability in modeling.

Data Balancing

Addressed class imbalance through upsampling of minority class within the training data.

Ensured balanced representation of target labels (0 and 1) before model training.

Model Training & Evaluation

Trained three supervised learning models:

Logistic Regression – Baseline model.

Random Forest Classifier – Captured non-linear relationships.

LightGBM – Gradient boosting for faster training and feature weighting.

Evaluated models using ROC curves, accuracy, and confusion matrices.

Observed test accuracy around 0.621, attributed to downsized dataset and reduced feature granularity.

Insights

Key predictive factors aligned with financial risk features.

Non-linear models (RF, LightGBM) outperformed logistic regression slightly but remained limited by data reduction.

ROC analysis revealed trade-offs between recall and false positive rates across models.

Model Persistence

Saved trained models using Joblib for future reuse (loan_default_model.pkl).

Tools

Python (Pandas, Scikit-learn, LightGBM, Matplotlib, Seaborn)

Google Colab for environment and runtime

Joblib for model serialization

