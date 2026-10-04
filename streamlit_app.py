"""
=============================================================================
AI-Based Loan Approval Prediction and Financial Risk Analysis System
Interactive Streamlit Dashboard
=============================================================================
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Set page config
st.set_page_config(
    page_title="AI Loan Approval & Risk Analysis System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "loan_model.pkl")
PREP_PATH = os.path.join(BASE_DIR, "model", "preprocessing.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "model", "metrics.json")
FEAT_IMP_PATH = os.path.join(BASE_DIR, "model", "feature_importance.json")

@st.cache_resource
def load_model_artifacts():
    model = joblib.load(MODEL_PATH)
    preprocessing = joblib.load(PREP_PATH)
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    with open(FEAT_IMP_PATH, "r", encoding="utf-8") as f:
        feature_importance = json.load(f)
    return model, preprocessing, metrics, feature_importance

try:
    model, preprocessing, metrics, feature_importance = load_model_artifacts()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}. Please run `python train_model.py` first.")
    st.stop()

# Header
st.title("🏦 AI-Based Loan Approval Prediction & Financial Risk Analysis")
st.markdown("Automated loan underwriting, credit score analysis, and risk assessment powered by Machine Learning.")

# Sidebar Navigation
mode = st.sidebar.radio("Navigation", ["🔮 Loan Predictor", "📊 Analytics & Model Comparison", "ℹ️ About System"])

if mode == "🔮 Loan Predictor":
    st.subheader("Applicant Financial & Demographic Profile")
    
    # Preset selection in sidebar
    preset = st.sidebar.selectbox(
        "⚡ Quick-Fill Preset Profiles",
        ["Custom", "Prime Tech Borrower (High Approval)", "Risky Credit History (High Rejection)", "Self-Employed Entrepreneur", "First-Time Starter Home"]
    )
    
    defaults = {
        "gender": "Male", "married": "Yes", "dependents": "0", "education": "Graduate",
        "self_employed": "No", "app_income": 5000, "coapp_income": 1500,
        "loan_amt": 150, "term": 360, "credit": "Good Credit History (1.0)", "property": "Urban"
    }
    
    if preset == "Prime Tech Borrower (High Approval)":
        defaults.update({"gender": "Male", "married": "Yes", "dependents": "1", "education": "Graduate", "app_income": 8500, "coapp_income": 3000, "loan_amt": 180, "credit": "Good Credit History (1.0)", "property": "Urban"})
    elif preset == "Risky Credit History (High Rejection)":
        defaults.update({"gender": "Male", "married": "No", "dependents": "2", "education": "Not Graduate", "app_income": 2200, "coapp_income": 0, "loan_amt": 220, "credit": "Poor / Default History (0.0)", "property": "Rural"})
    elif preset == "Self-Employed Entrepreneur":
        defaults.update({"gender": "Female", "married": "Yes", "self_employed": "Yes", "app_income": 7500, "coapp_income": 4500, "loan_amt": 250, "term": 240, "credit": "Good Credit History (1.0)", "property": "Semiurban"})
    elif preset == "First-Time Starter Home":
        defaults.update({"gender": "Male", "married": "No", "dependents": "0", "app_income": 4200, "coapp_income": 1200, "loan_amt": 95, "term": 180, "credit": "Good Credit History (1.0)", "property": "Semiurban"})

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Personal & Employment Details")
        gender = st.selectbox("Gender", ["Male", "Female"], index=0 if defaults["gender"]=="Male" else 1)
        married = st.selectbox("Married Status", ["Yes", "No"], index=0 if defaults["married"]=="Yes" else 1)
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"], index=["0", "1", "2", "3+"].index(defaults["dependents"]))
        education = st.selectbox("Education", ["Graduate", "Not Graduate"], index=0 if defaults["education"]=="Graduate" else 1)
        self_employed = st.selectbox("Self Employed", ["No", "Yes"], index=0 if defaults["self_employed"]=="No" else 1)
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"], index=["Urban", "Semiurban", "Rural"].index(defaults["property"]))

    with col2:
        st.markdown("#### Financial & Credit History")
        applicant_income = st.number_input("Applicant Income ($/month)", min_value=0, value=int(defaults["app_income"]), step=100)
        coapplicant_income = st.number_input("Co-Applicant Income ($/month)", min_value=0, value=int(defaults["coapp_income"]), step=100)
        loan_amount = st.number_input("Requested Loan Amount (in $ thousands, e.g. 150 = $150k)", min_value=10, max_value=1000, value=int(defaults["loan_amt"]), step=5)
        loan_term = st.selectbox("Loan Repayment Term (months)", [120, 180, 240, 360, 480], index=[120, 180, 240, 360, 480].index(defaults["term"]))
        credit_history_label = st.selectbox("Credit History", ["Good Credit History (1.0)", "Poor / Default History (0.0)"], index=0 if "1.0" in defaults["credit"] else 1)
        credit_history = 1.0 if "1.0" in credit_history_label else 0.0

    # Real-time metrics bar
    total_income = applicant_income + coapplicant_income
    emi = (loan_amount * 1000) / loan_term
    balance_income = total_income - emi
    dti = (emi / total_income * 100) if total_income > 0 else 0

    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Household Income", f"${total_income:,.0f}")
    m2.metric("Estimated Monthly EMI", f"${emi:,.0f}/mo")
    m3.metric("Debt-to-Income (DTI)", f"{dti:.1f}%")
    m4.metric("Net Balance Income", f"${balance_income:,.0f}")

    if st.button("🔮 Predict Loan Approval", type="primary", use_container_width=True):
        # Preprocessing
        encoders = preprocessing['encoders']
        feature_cols = preprocessing['feature_cols']

        input_row = {
            'Gender': encoders['Gender']['mapping'].get(gender, 1),
            'Married': encoders['Married']['mapping'].get(married, 1),
            'Dependents': encoders['Dependents']['mapping'].get(dependents, 0),
            'Education': encoders['Education']['mapping'].get(education, 0),
            'Self_Employed': encoders['Self_Employed']['mapping'].get(self_employed, 0),
            'ApplicantIncome': applicant_income,
            'CoapplicantIncome': coapplicant_income,
            'LoanAmount': loan_amount,
            'Loan_Amount_Term': loan_term,
            'Credit_History': credit_history,
            'Property_Area': encoders['Property_Area']['mapping'].get(property_area, 2),
            'TotalIncome': total_income,
            'EMI': emi,
            'BalanceIncome': balance_income
        }

        input_df = pd.DataFrame([input_row])[feature_cols]
        pred = model.predict(input_df)[0]
        probs = model.predict_proba(input_df)[0]
        prob_reject, prob_approve = float(probs[0]), float(probs[1])
        confidence = max(prob_approve, prob_reject) * 100

        st.markdown("---")
        st.subheader("Prediction Adjudication Result")

        r_col1, r_col2 = st.columns([1, 1.2])

        with r_col1:
            if pred == 1:
                st.success("### 🟢 LOAN APPROVED")
                st.markdown(f"**Approval Confidence:** `{confidence:.1f}%`")
                st.progress(prob_approve)
                risk_level = "Low Risk" if prob_approve >= 0.75 else "Moderate Risk"
                st.info(f"**Financial Risk Tier:** `{risk_level}`")
            else:
                st.error("### 🔴 LOAN NOT APPROVED")
                st.markdown(f"**Rejection Confidence:** `{confidence:.1f}%`")
                st.progress(prob_reject)
                st.warning("**Financial Risk Tier:** `High Default Risk`")

        with r_col2:
            st.markdown("#### Key Driving Underwriting Factors")
            if credit_history == 1.0:
                st.markdown("✅ **Credit History (Positive):** Clean repayment history satisfies credit standards.")
            else:
                st.markdown("❌ **Credit History (Negative):** Historical delinquency strongly correlates with default risk.")

            if dti <= 35:
                st.markdown(f"✅ **Debt-to-Income (Positive):** Healthy DTI of {dti:.1f}% leaves substantial disposable income.")
            elif dti <= 50:
                st.markdown(f"⚠️ **Debt-to-Income (Moderate):** DTI of {dti:.1f}% approaches institutional debt ceiling.")
            else:
                st.markdown(f"❌ **Debt-to-Income (Negative):** Elevated DTI of {dti:.1f}% poses repayment strain.")

            if total_income >= 5000:
                st.markdown(f"✅ **Household Income (Positive):** Combined income of ${total_income:,.0f} meets minimum threshold.")
            else:
                st.markdown(f"⚠️ **Household Income (Moderate):** Combined income of ${total_income:,.0f} limits higher credit brackets.")

elif mode == "📊 Analytics & Model Comparison":
    st.subheader("Model Comparison & Underwriting Analytics")
    
    # Leaderboard table
    rows = []
    for m_name, vals in metrics.items():
        rows.append({
            "Algorithm": m_name,
            "Accuracy": f"{vals['accuracy']*100:.2f}%",
            "Precision": f"{vals['precision']*100:.2f}%",
            "Recall": f"{vals['recall']*100:.2f}%",
            "F1-Score": f"{vals['f1_score']*100:.2f}%",
            "ROC-AUC": f"{vals['roc_auc']*100:.2f}%",
            "5-Fold CV Accuracy": f"{vals['cv_mean']*100:.2f}% (±{vals['cv_std']*100:.1f}%)"
        })
    df_metrics = pd.DataFrame(rows)
    st.dataframe(df_metrics, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Feature Importance Ranking")
        feat_df = pd.DataFrame(list(feature_importance.items())[:8], columns=["Feature", "Importance"])
        feat_df["Importance (%)"] = feat_df["Importance"] * 100
        st.bar_chart(feat_df.set_index("Feature")["Importance (%)"])

    with c2:
        st.markdown("#### Best Model (Random Forest) Confusion Matrix")
        rf_cm = metrics['Random Forest']['confusion_matrix']
        cm_df = pd.DataFrame(
            rf_cm,
            index=["Actual Rejected (N)", "Actual Approved (Y)"],
            columns=["Predicted Rejected", "Predicted Approved"]
        )
        st.table(cm_df)

    st.markdown("---")
    st.markdown("#### Exploratory Data Analysis (EDA) Highlights")
    eda_col1, eda_col2 = st.columns(2)
    
    img_status = os.path.join(BASE_DIR, "static", "images", "eda_loan_status.png")
    img_credit = os.path.join(BASE_DIR, "static", "images", "eda_credit_history.png")
    img_model = os.path.join(BASE_DIR, "static", "images", "eda_model_comparison.png")
    img_income = os.path.join(BASE_DIR, "static", "images", "eda_income_dist.png")

    if os.path.exists(img_status):
        eda_col1.image(img_status, caption="Overall Loan Approval Distribution", use_container_width=True)
    if os.path.exists(img_credit):
        eda_col2.image(img_credit, caption="Credit History vs Loan Approval", use_container_width=True)
    if os.path.exists(img_income):
        eda_col1.image(img_income, caption="Income Distributions & Loan Amounts", use_container_width=True)
    if os.path.exists(img_model):
        eda_col2.image(img_model, caption="Multi-Model Benchmark", use_container_width=True)

else:
    st.subheader("System Architecture & Problem Formulation")
    st.markdown("""
    ### AI-Based Loan Approval Prediction and Financial Risk Analysis System
    This project automates credit evaluation in the fintech sector using machine learning.
    
    **Workflow:**
    1. **Data Intake & Preprocessing**: Missing value imputation with median (for skewed numericals) and mode (for categoricals).
    2. **Feature Engineering**: Calculation of Total Household Income, Equated Monthly Installment (EMI), and Net Disposable Income.
    3. **Ensemble Modeling**: Multi-algorithm evaluation across Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting.
    4. **Underwriting Intelligence**: Decision output with confidence percentage, risk tiers, and key factor explainability.
    
    Developed as an end-to-end Machine Learning Internship Project.
    """)
