
# Loan Default Prediction with Machine Learning

This project explores **predicting loan defaults** using machine learning models.  
The dataset contains borrower information (loan amount, interest rate, annual income, credit history, home ownership, state, etc.) with a binary target variable:  
- `0 = fully paid`  
- `1 = charged off`  

---

## 🔑 Key Steps

### Data Preprocessing
- Scaled numerical features  
- Handled categorical features with one-hot encoding  
- Addressed class imbalance with upsampling  

### Model Training & Evaluation
- **Logistic Regression** (baseline)  
- **Random Forest Classifier** (improved performance, captured non-linear relationships)  
- Compared accuracy, ROC curves, and feature importance  

### Insights
- Features like interest rate (`int_rate`), grade, and term length were most predictive  
- Location (state) features had little contribution  
- ROC curve analysis showed trade-offs between recall and false positive rate  

### Model Persistence
- Final Random Forest model saved with `joblib` for future predictions (`loan_default_model.pkl`)  

---

## 🚀 Next Steps
- Deploy as a simple web API to predict default risk for new applicants  
- Experiment with gradient boosting models (XGBoost, LightGBM) for further improvements  

---

## 📂 Repository Structure (suggested)
