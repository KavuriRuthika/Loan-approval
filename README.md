# AI-Based Loan Approval Prediction and Financial Risk Analysis System

An end-to-end Machine Learning and Credit Underwriting Intelligence system developed using **Python, Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn, Flask, HTML/CSS/JavaScript, and Streamlit**.

---

## 📌 1. Project Overview & Problem Statement

### The Problem
In the modern fintech and banking sector, evaluating loan applications manually is:
- **Time-consuming and labor-intensive:** Human underwriters must inspect multi-attribute financial statements and credit bureau filings.
- **Subjective & prone to inconsistent credit policy:** Different loan officers may evaluate identical risk parameters inconsistently.
- **Vulnerable to default exposure:** Without multivariate statistical models, hidden interaction risks (e.g., high debt-to-income coupled with marginal credit records) are overlooked.

### The Solution
This system analyzes historical credit applicant data, executes automated data cleaning and financial feature engineering, benchmarks multiple classification algorithms (Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting), and deploys the champion model into an interactive web portal featuring real-time risk tiering, Equated Monthly Installment (EMI) stress testing, Debt-to-Income (DTI) metrics, and explainable AI factor breakdowns.

---

## 🏗️ 2. System Architecture & Workflow

```
                         ┌────────────────────────────────────────┐
                         │   Historical Loan Dataset (CSV)        │
                         └───────────────────┬────────────────────┘
                                             │
                                             ▼
                         ┌────────────────────────────────────────┐
                         │  Missing Value Imputation & Cleaning   │
                         │  (Median for Skewed, Mode for Nominal) │
                         └───────────────────┬────────────────────┘
                                             │
                                             ▼
                         ┌────────────────────────────────────────┐
                         │       Financial Feature Engineering    │
                         │   - Total Household Income             │
                         │   - Estimated Monthly EMI              │
                         │   - Net Disposable Balance Income      │
                         │   - Debt-to-Income (DTI) Ratio         │
                         └───────────────────┬────────────────────┘
                                             │
                                             ▼
                         ┌────────────────────────────────────────┐
                         │  Label Encoding & Train/Test Split     │
                         │  (80% Train, 20% Stratified Test Split)│
                         └───────────────────┬────────────────────┘
                                             │
                                             ▼
                         ┌────────────────────────────────────────┐
                         │     Multi-Algorithm Benchmarking       │
                         │  - Logistic Regression (with Scaler)   │
                         │  - Decision Tree Classifier            │
                         │  - Random Forest Classifier (Champion) │
                         │  - Gradient Boosting Classifier        │
                         └───────────────────┬────────────────────┘
                                             │
                                             ▼
                         ┌────────────────────────────────────────┐
                         │  Model Evaluation & Serialization      │
                         │  (Accuracy, F1, AUC, 5-Fold Strat CV)  │
                         │  Dump to loan_model.pkl & prep.pkl     │
                         └───────────────────┬────────────────────┘
                                             │
                     ┌───────────────────────┴───────────────────────┐
                     ▼                                               ▼
     ┌───────────────────────────────┐               ┌───────────────────────────────┐
     │  Full-Stack Flask Web App     │               │  Interactive Streamlit App    │
     │  - REST API (/api/predict)    │               │  streamlit run                │
     │  - Modern Fintech Glassmorphism│              │  streamlit_app.py             │
     │  - 1-Click Preset Profiles    │               │                               │
     │  - Real-time Calculations     │               │                               │
     └───────────────────────────────┘               └───────────────────────────────┘
```

---

## 📊 3. Dataset Description

The system trains on the standard fintech loan credit dataset (`614 rows × 13 attributes`):

| Feature Name | Type | Description | Values / Examples |
|---|---|---|---|
| `Loan_ID` | String | Unique Application Identifier | Dropped during modeling |
| `Gender` | Categorical | Applicant Gender | `Male`, `Female` |
| `Married` | Categorical | Marital Status | `Yes`, `No` |
| `Dependents` | Categorical | Number of financial dependents | `0`, `1`, `2`, `3+` |
| `Education` | Categorical | Academic Qualification | `Graduate`, `Not Graduate` |
| `Self_Employed` | Categorical | Employment arrangement | `Yes`, `No` |
| `ApplicantIncome`| Numerical | Primary applicant monthly income | e.g. `$5,000` |
| `CoapplicantIncome`| Numerical | Co-applicant monthly income | e.g. `$1,500` |
| `LoanAmount` | Numerical | Requested loan principal | In thousands (e.g. `150` = `$150,000`) |
| `Loan_Amount_Term`| Numerical | Loan amortization duration | In months (`120`, `180`, `240`, `360`, `480`) |
| `Credit_History` | Categorical | Credit bureau repayment record | `1.0` (Good/Clean), `0.0` (Default/Poor) |
| `Property_Area` | Categorical | Geographic sector of property | `Urban`, `Semiurban`, `Rural` |
| **`Loan_Status`** | **Target** | **Underwriting Decision** | **`Y` (Approved)**, **`N` (Rejected)** |

---

## 🔬 4. Machine Learning Model Leaderboard

Evaluated on an independent 20% stratified holdout test set (`n=123`) and validated using **5-Fold Stratified Cross-Validation**:

| Algorithm | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC | 5-Fold CV Accuracy |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest (Champion)** | **84.55%** | **83.67%** | **96.47%** | **89.62%** | **0.8322** | **79.83% (±2.5%)** |
| Logistic Regression | 86.18% | 84.00% | 98.82% | 90.81% | 0.8025 | 79.83% (±2.8%) |
| Gradient Boosting | 82.93% | 82.00% | 96.47% | 88.65% | 0.7700 | 78.60% (±3.1%) |
| Decision Tree | 81.30% | 82.29% | 92.94% | 87.29% | 0.7856 | 76.98% (±3.4%) |

### Why Random Forest is the Production Champion:
1. **High Discrimination (ROC-AUC 0.8322):** Delivers the highest area under the ROC curve, guaranteeing optimal rank-ordering of credit risk.
2. **High Recall (96.47%):** Minimizes false negatives, ensuring qualified creditworthy applicants are not erroneously turned away.
3. **Resilience to Overfitting:** The ensemble bagging mechanism aggregates 250 de-correlated decision trees with controlled max depth and minimum leaf samples.

---

## 🎯 5. Feature Importance Breakdown

Random Forest Gini Impurity ranking reveals the real-world financial drivers:

```
Credit_History        ████████████████████ 43.69%
BalanceIncome         ████ 9.06%
TotalIncome           ████ 8.41%
EMI                   ████ 8.13%
ApplicantIncome       ███  7.77%
LoanAmount            ███  7.18%
CoapplicantIncome     ██   5.59%
Property_Area         █    2.78%
Loan_Amount_Term      █    2.23%
Dependents            █    1.92%
Married                    1.19%
Education                  1.11%
Self_Employed              0.51%
Gender                     0.43%
```

---

## 📁 6. Repository Folder Structure

```
LoanApprovalPrediction/
│
├── dataset/
│   └── loan_data.csv                    # Historical loan dataset (614 records)
│
├── model/
│   ├── loan_model.pkl                   # Trained Random Forest Champion model
│   ├── preprocessing.pkl                # Scalers, imputers, and categorical mappings
│   ├── scaler.pkl                       # StandardScaler for linear algorithms
│   ├── metrics.json                     # Complete performance benchmark data
│   └── feature_importance.json          # Gini feature weights
│
├── notebooks/
│   └── analysis.ipynb                   # Complete Jupyter Notebook with EDA & modeling
│
├── static/
│   ├── css/
│   │   └── style.css                    # Modern Fintech Glassmorphism UI stylesheet
│   ├── js/
│   │   └── script.js                    # 1-Click Presets & Real-Time Financial Calculators
│   └── images/                          # Publication-quality EDA & Benchmark figures
│       ├── eda_loan_status.png
│       ├── eda_credit_history.png
│       ├── eda_property_area.png
│       ├── eda_education.png
│       ├── eda_income_dist.png
│       ├── eda_correlation.png
│       ├── eda_model_comparison.png
│       └── eda_feature_importance.png
│
├── templates/
│   ├── index.html                       # Executive Portal & System Architecture
│   ├── prediction.html                  # Interactive Credit Risk Simulator & Results
│   └── dashboard.html                   # Analytics, Model Benchmark & EDA Hub
│
├── train_model.py                       # Automated ML training and serialization script
├── app.py                              # Flask Web Server & JSON REST API
├── streamlit_app.py                    # Streamlit Dashboard alternative
├── requirements.txt                    # Project dependency specification
├── README.md                           # Documentation & Viva walkthrough
└── .gitignore                          # Standard gitignore
```

---

## 🚀 7. Installation & Quick Start Guide

### Step 1: Clone or Navigate to the Directory
```bash
cd r:\MLintership2
```

### Step 2: (Optional) Create and Activate a Virtual Environment
```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# macOS / Linux:
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Training Pipeline (Re-trains & Generates Graphics)
```bash
python train_model.py
```

### Step 5: Launch the Web Application

#### Option A: Run Full Flask Web Portal (Recommended for Evaluation & Resume)
```bash
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```
- **Home:** `http://127.0.0.1:5000/`
- **Predictor:** `http://127.0.0.1:5000/predict`
- **Dashboard & Leaderboard:** `http://127.0.0.1:5000/dashboard`

#### Option B: Run Streamlit Application
```bash
streamlit run streamlit_app.py
```

---

## 🔌 8. REST API Reference

The Flask application exposes a production-ready JSON REST API endpoint for third-party or mobile integrations:

### `POST /api/predict`
**Request Headers:** `Content-Type: application/json`

**Sample Request Body:**
```json
{
  "Gender": "Male",
  "Married": "Yes",
  "Dependents": "1",
  "Education": "Graduate",
  "Self_Employed": "No",
  "ApplicantIncome": 8500,
  "CoapplicantIncome": 3000,
  "LoanAmount": 180,
  "Loan_Amount_Term": 360,
  "Credit_History": "1.0",
  "Property_Area": "Urban"
}
```

**Sample Response Body:**
```json
{
  "approval_probability": 76.2,
  "balance_income": 11000.0,
  "confidence": 76.2,
  "dti_ratio": 4.3,
  "factors": [
    {
      "detail": "Meets credit standards with clear historical repayment history.",
      "factor": "Credit History",
      "impact": "Positive"
    },
    {
      "detail": "Healthy DTI of 4.3%; adequate monthly disposable income.",
      "factor": "Debt-to-Income (DTI)",
      "impact": "Positive"
    },
    {
      "detail": "Strong combined income of $11,500.",
      "factor": "Household Total Income",
      "impact": "Positive"
    }
  ],
  "is_approved": true,
  "monthly_emi": 500.0,
  "rejection_probability": 23.8,
  "risk_color": "success",
  "risk_level": "Low Risk",
  "status": "APPROVED",
  "total_income": 11500.0
}
```

---

## 🎓 9. Viva / Faculty Demonstration Walkthrough

When presenting this project to your faculty, evaluator, or technical interviewer, follow this 10-step demonstration sequence:

1. **Step 1: Open Home Portal (`/`):**
   - Introduce the title: *"AI-Based Loan Approval Prediction and Financial Risk Analysis System"*.
   - Point out the 4 KPI cards (Random Forest champion, 86.2% peak accuracy, 614 historical applicants, 68.7% baseline approval).
2. **Step 2: Explain Architecture:**
   - Walk through the 4-step underwriting pipeline: Intake -> Financial Stress Metrics -> ML Inference -> Explainability Audit.
3. **Step 3: Navigate to Analytics & Model Benchmark (`/dashboard`):**
   - Show the **Algorithm Leaderboard**: Compare Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting across Accuracy, F1-Score, Precision, Recall, and 5-Fold Stratified Cross-Validation.
   - Show the **Confusion Matrix Breakdown**: Detail True Positives, True Negatives, Type I (False Positive), and Type II (False Negative) errors.
4. **Step 4: Show the Exploratory Data Analysis (EDA) Gallery:**
   - Explain why **Credit History** has a 43.7% feature importance weight (applicants with 0.0 default history face a >92% rejection rate).
   - Explain the influence of **Property Area** (Semiurban properties enjoy higher approval rates).
   - Explain how **Total Household Income** and **Debt-to-Income (DTI)** normalize income skewness.
5. **Step 5: Navigate to Loan Simulator (`/predict`):**
   - Click the fast-fill preset: **"Prime Corporate (High Approval)"**.
   - Show how the real-time financial ribbon instantly calculates Total Household Income ($11,500), Estimated Monthly EMI ($500/mo), and a healthy DTI of 4.3%.
   - Click **"Predict Loan Approval"**.
   - Show the output: **🟢 LOAN APPROVED**, Confidence: `~76.2%`, Risk Tier: `Low Risk`, and explainability bullet points.
6. **Step 6: Demonstrate Risk Adjudication:**
   - Click the preset: **"Default Risk (High Rejection)"** (Credit History: 0.0, low income, high requested principal).
   - Click **"Predict Loan Approval"**.
   - Show the output: **🔴 LOAN NOT APPROVED**, Confidence: `~79.3%`, Risk Tier: `High Default Risk`, with adverse action factors highlighted.
7. **Step 7: Code Walkthrough (`train_model.py` and `analysis.ipynb`):**
   - Point out the median/mode imputation, feature engineering (`EMI`, `TotalIncome`, `BalanceIncome`), stratified train/test split, and model persistence using `joblib`.
8. **Step 8: REST API Demo:**
   - Mention the programmatic endpoint `/api/predict` allowing mobile apps or bank core banking software to consume the model via JSON.

---

## 📜 10. License & Acknowledgments
- Dataset provided as part of the Machine Learning Internship Program.
- Developed with best practices in ethical AI, reproducible data science, and financial credit risk engineering.
