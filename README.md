
# Loan Default Prediction with Machine Learning

This project explores **predicting loan defaults** using machine learning models.  
The dataset contains borrower information (loan amount, interest rate, annual income, credit history, home ownership, state, etc.) with a binary target variable:  
- `0 = fully paid`  
- `1 = charged off`  

---

# 🧠 Loan Default Prediction

A machine learning project predicting **loan default risk** using financial and demographic data.  
Built and tested on **Google Colab** under limited memory constraints (downsized dataset: 200,000 rows).  
This project demonstrates the full ML pipeline — from data merging and cleaning to model training and evaluation.

---

## 📁 Project Overview

This project explores the impact of borrower and loan attributes on default probability.  
Multiple CSV datasets were merged (via **one-to-many relationships**) and preprocessed for model development.  

---

## ⚙️ Data Preprocessing

1. **Data Cleaning**
   - Removed missing and null values.  
   - Standardized inconsistent column names and formats.  

2. **Feature Engineering**
   - Split features into **categorical** and **numerical** types.  
   - Applied **one-hot encoding** to categorical variables.  
   - Scaled numerical features for consistency.  
   - Created **binned features** for interpretability (e.g., income bands, credit score ranges).  

3. **Dataset Balancing**
   - Addressed target imbalance by **upsampling** minority class (default = 1).  
   - Ensured balanced representation between defaulters and non-defaulters.  

4. **Train-Test Split**
   - 80% Training | 20% Testing  

---

## 🤖 Model Training & Evaluation

| Model | Description | Notes |
|:------|:-------------|:------|
| **Logistic Regression** | Baseline linear model | Fast, interpretable |
| **Random Forest** | Ensemble method capturing non-linear relationships | Moderate accuracy |
| **LightGBM** | Gradient boosting model for large-scale data | Fast and efficient |

### Evaluation Metrics
- **Accuracy**
- **ROC Curve**
- **Confusion Matrix**
- **Feature Importance**

Average **test accuracy ≈ 0.621**, primarily limited by downsized data and reduced granularity.

---

## 📊 Insights

- Key predictive features included **interest rate, grade, and term length**.  
- Geographic/location features contributed little predictive value.  
- **ROC curve analysis** revealed trade-offs between recall and false positive rates.  
- Random Forest and LightGBM outperformed Logistic Regression slightly.  

---

## 💾 Model Persistence

Final model saved using **Joblib** for future use:

```python
import joblib

# Save model
joblib.dump(rf_model, 'loan_default_model.pkl')

# Load model
loaded_model = joblib.load('loan_default_model.pkl')


