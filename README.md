# Fraudulent_transactions_detection_model
Fraudulent_transactions_detection_model
## Project Overview
This project aims to detect fraudulent financial transactions using **machine learning models** and visualize the results using Python. The analysis is based on a large-scale transaction dataset containing over 6 million records. Two primary models were implemented: **Logistic Regression** and **Random Forest Classifier**.

**Key Goals:**
- Identify fraudulent transactions with high recall.
- Understand which features contribute most to fraud detection.
- Provide actionable insights for risk mitigation.

---

## Motivation
Financial fraud is a growing concern for banks and digital payment platforms. Detecting fraud early is crucial to minimize losses and protect customers. This project combines data preprocessing, exploratory analysis, and machine learning to build an efficient fraud detection system.

---

## Dataset
- Source: Synthetic financial transaction dataset (CSV format)
- Size: 6,362,620 records, 11 columns
- Key Features:
  - `step`: Time step of transaction
  - `amount`: Transaction amount
  - `oldbalanceOrg` & `newbalanceOrig`: Sender account balances
  - `oldbalanceDest` & `newbalanceDest`: Recipient account balances
  - `type`: Transaction type (Payment, Transfer, Cash Out, Debit)
  - `isFraud`: Target variable (1 = Fraud, 0 = Legitimate)

---

## Tools & Libraries
- **Python:** Pandas, NumPy, Scikit-learn, Imbalanced-learn  
- **Visualization:** Matplotlib, Seaborn, Plotly, Dash  
- **Modeling:** Logistic Regression, Random Forest  
- **Other:** Jupyter Notebook  

---

## Data Preprocessing
- Handled categorical variables using one-hot encoding for `type`.
- Removed high-cardinality features `nameOrig` and `nameDest`.
- Scaled numeric features using `StandardScaler`.
- Split dataset into training (80%) and testing (20%) subsets.
- Handled class imbalance using class weighting.

---

## Modeling

### Logistic Regression
- Pipeline with StandardScaler + Logistic Regression
- Hyperparameter tuning with GridSearchCV using **F1-score** as metric
- **Best parameters:** `C=100`, `penalty='l1'`
- **Best F1-score (CV):** 0.67
- Key predictive features: `amount`, `oldbalanceOrg`, `newbalanceOrig`

### Random Forest Classifier
- Pipeline with StandardScaler + Random Forest
- Hyperparameter tuning with GridSearchCV using **Recall** as metric
- **Best parameters:** `n_estimators=100`, `max_depth=10`, `class_weight='balanced'`
- **Best CV Recall Score:** 0.657
- **AUC-ROC Score:** 0.972
- Important features: `amount`, `oldbalanceOrg`, `newbalanceOrig`, `type_TRANSFER`, `step`

---

## Model Evaluation
- Confusion matrices for both models visualized using Seaborn & Matplotlib
- Precision-Recall analysis used to select an optimal threshold (0.48) for Random Forest
- Observed trade-off between precision and recall to minimize false negatives

---

## Key Insights
- Large transactions from accounts with high balances are more likely to be fraudulent.
- Transfer operations and unusual transaction timing (`step`) increase fraud risk.
- Both Logistic Regression and Random Forest identified similar key predictive features.
- High accuracy is misleading due to class imbalance; recall and AUC-ROC provide better evaluation.

---

## Deployment & Future Work
- Could be integrated into real-time transaction systems for **fraud scoring**.
- Threshold-based triggers, velocity checks, and geo/device anomaly detection recommended.
- Monitor model performance over time and adjust thresholds for evolving fraud patterns.
- Increase training dataset size to improve model performance and reduce false negatives.
