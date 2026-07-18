🏦 First Payment Default (FPD) Prediction — Microfinance Lending

Predicting whether a borrower will default on their first loan installment within a 5-day delinquency window, using a live cooperative banking PostgreSQL database.

📋 Project Overview

This project develops a machine learning pipeline to predict First Payment Default (FPD) in microfinance and cooperative banking lending.

FPD is defined as failure to pay the first loan installment within 5 days of the due date. It is the earliest and strongest signal of loan default risk, making it a critical metric for credit underwriting.

Item                 Detail
Institution Type     Cooperative Bank / Microfinance (SHG/JLG Group Lending)
Database             PostgreSQL — tenant_prabodh
Tables Used          15 out of 69 domain tables
Total Records        4,000 loans (after FPD filter) 
Target               FPD = 1 (Default) / FPD = 0 (No Default) 
Class Distribution   80% No Default / 20% Default 
Best Model           XGBoost — ROC-AUC: 0.6486

📁 Repository Structure

fpd-prediction/
│
├── main.py                        # Full pipeline — data extraction to model training
│
├── results/
│   ├── output.txt                 # Output of the code
│   ├── Confusion_matrix.png       # Confusion matrix — XGBoost
|   ├── ROC_AUC.png                # Bar chart — all models ROC-AUC  
│   └── XGBoost.png                # Top 20 features — XGBoost
│
├── FPD_Research_Paper.docx        # Full research paper
└── README.md                      # This file


🎯 Target Variable Construction(FPD

* Select the **first installment** for each loan.
* Set a **5-day grace period** from the installment due date.
* Consider only loans where the 5-day period has already ended.
* Assign:
  * **FPD = 1 (Default):** Payment not made or made after the 5-day deadline.
  * **FPD = 0 (No Default):** Payment made within the 5-day deadline.


🗂️ Domain Features

Domain                   Features          Top Feature
Prior Loan History       10                max_par30_amount
Members                  8                 member_age
Loan Product             7                 sanctioned_amount
Savings Behavior         8                 savings_to_loan_ratio
Groups                   8                 group_default_rate
KYC Compliance           7                 kyc_completed_before_loan
Guarantor Exposure       6                 avg_guarantor_burden_tier  
Branchs                  6                 branch_avg_sanctioned_amount


📊 Model Results

Model                ROC-AUC    Somers D    Accuracy
XGBoost              0.6486     0.2972      78.25%
Logistic Regression  0.6171     0.2343      80.00%
Random Fores t       0.6071     0.2142      74.75%
Decision Tree        0.5258     0.0516      69.12%

Best Model: XGBoost with ROC-AUC 0.6486 and Somers D 0.2972


🔧 Setup & Usage

Prerequisites

bashpip install sqlalchemy psycopg2-binary pandas numpy scikit-learn xgboost tensorflow matplotlib

Database Connection

pythonfrom sqlalchemy import create_engine

engine = create_engine("postgresql+psycopg2://analyst:analyst123@49.207.11.47:6432/tenant_prabodh")

Run Pipeline
# Run the full pipeline
python main.py

⚠️ Requires active database connection to tenant_prabodh PostgreSQL server.


🏗️ Pipeline Architecture

Database (PostgreSQL)
        │
        ▼
Load Tables (SQLAlchemy)
        │
        ▼
Create FPD Target (5-day window)
        │
        ▼
Feature Engineering (8 Domains → 70+ features)
        │
        ▼
Preprocessing
├── Null imputation (median / Unknown)
├── Log transformation (skewed features)
├── One-hot encoding (categorical)
└── Feature selection (19 final features)
        │
        ▼
Train-Test Split (80:20, stratified)
        │
        ▼
Model Training
├── Logistic Regression
├── Decision Tree
├── Random Forest
├── XGBoost
└── ANN (TensorFlow/Keras)
        │
        ▼
Evaluation
├── ROC-AUC
├── Somers D
├── Accuracy
├── Confusion Matrix
└── Feature Importance


📈 Confusion Matrix — XGBoost

                Predicted: No Default    Predicted: Default
Actual: No Default       617                    23
Actual: Default          151                     9


True Negatives (Correctly identified No Default): 617
True Positives (Correctly identified Default): 9
False Negatives (Missed Defaults): 151 ← key risk
False Positives (False Alarms): 23



💡 Key Findings

1. Prior repayment history dominates — PAR30 and overdue bucket features from prior loans are by far the strongest predictors
2. First-time borrowers are harder to predict — they have no loan history features (all zeros), requiring behavioral proxies from savings and group attendance
3. Group risk matters — group_default_rate at rank 14 confirms that collective credit culture affects individual default risk
4. Demographics carry signal — age and marital status reflect income stability and financial vulnerability
5. 5-day window is strict — extending to 30 days would likely improve model performance



👤 Author

Pobbathi Sai Ganesh
B.Tech Computer Science and Engineering — Year III
National Institute of Technology Goa

Guide: Mr. Jagat Chaitanya Prabhala & Vikas Putcha


📄 License

This project is developed as part of an academic research exercise on microfinance credit risk modeling.
