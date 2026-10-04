"""
=============================================================================
AI-Based Loan Approval Prediction and Financial Risk Analysis System
Flask Web Application & REST API
=============================================================================
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "loan_model.pkl")
PREP_PATH = os.path.join(BASE_DIR, "model", "preprocessing.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "model", "metrics.json")
FEAT_IMP_PATH = os.path.join(BASE_DIR, "model", "feature_importance.json")
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "loan_data.csv")

# Load serialized model & preprocessing pipeline
model = None
preprocessing = None
metrics_data = {}
feature_importance_data = {}

def load_artifacts():
    global model, preprocessing, metrics_data, feature_importance_data
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
        if os.path.exists(PREP_PATH):
            preprocessing = joblib.load(PREP_PATH)
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
        if os.path.exists(FEAT_IMP_PATH):
            with open(FEAT_IMP_PATH, "r", encoding="utf-8") as f:
                feature_importance_data = json.load(f)
        print("Model and preprocessing artifacts successfully loaded.")
    except Exception as e:
        print(f"Error loading artifacts: {e}")

load_artifacts()

def compute_prediction(data):
    """
    Validate, preprocess, and predict loan approval with risk score.
    Expects dictionary of raw applicant inputs.
    """
    encoders = preprocessing['encoders']
    feature_cols = preprocessing['feature_cols']

    # Extract numerical inputs with sensible defaults
    applicant_income = float(data.get('ApplicantIncome', 0))
    coapplicant_income = float(data.get('CoapplicantIncome', 0))
    loan_amount = float(data.get('LoanAmount', 100)) # in thousands ($)
    loan_term = float(data.get('Loan_Amount_Term', 360)) # in months
    credit_history = float(data.get('Credit_History', 1.0))

    # Feature engineering
    total_income = applicant_income + coapplicant_income
    emi = (loan_amount * 1000) / (loan_term if loan_term > 0 else 360)
    balance_income = total_income - emi
    dti_ratio = (emi / total_income * 100) if total_income > 0 else 100.0

    # Encode categorical fields
    gender = encoders['Gender']['mapping'].get(str(data.get('Gender', 'Male')), 1)
    married = encoders['Married']['mapping'].get(str(data.get('Married', 'Yes')), 1)
    dependents = encoders['Dependents']['mapping'].get(str(data.get('Dependents', '0')), 0)
    education = encoders['Education']['mapping'].get(str(data.get('Education', 'Graduate')), 0)
    self_employed = encoders['Self_Employed']['mapping'].get(str(data.get('Self_Employed', 'No')), 0)
    property_area = encoders['Property_Area']['mapping'].get(str(data.get('Property_Area', 'Urban')), 2)

    # Build input DataFrame
    input_row = {
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_term,
        'Credit_History': credit_history,
        'Property_Area': property_area,
        'TotalIncome': total_income,
        'EMI': emi,
        'BalanceIncome': balance_income
    }

    input_df = pd.DataFrame([input_row])[feature_cols]

    # Model inference
    pred = int(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0]
    prob_rejected = float(probabilities[0])
    prob_approved = float(probabilities[1])

    # Status, confidence, and risk rating
    status = "APPROVED" if pred == 1 else "REJECTED"
    confidence = round(max(prob_approved, prob_rejected) * 100, 1)
    approval_prob = round(prob_approved * 100, 1)

    if approval_prob >= 75:
        risk_level = "Low Risk"
        risk_color = "success"
    elif approval_prob >= 45:
        risk_level = "Moderate Risk"
        risk_color = "warning"
    else:
        risk_level = "High Risk"
        risk_color = "danger"

    # Explainability factors
    factors = []
    if credit_history == 1.0:
        factors.append({"factor": "Credit History", "impact": "Positive", "detail": "Meets credit standards with clear historical repayment history."})
    else:
        factors.append({"factor": "Credit History", "impact": "Negative", "detail": "Prior default or non-compliance significantly escalates credit risk."})

    if dti_ratio <= 35:
        factors.append({"factor": "Debt-to-Income (DTI)", "impact": "Positive", "detail": f"Healthy DTI of {dti_ratio:.1f}%; adequate monthly disposable income."})
    elif dti_ratio <= 50:
        factors.append({"factor": "Debt-to-Income (DTI)", "impact": "Neutral", "detail": f"DTI is {dti_ratio:.1f}%; close to standard banking underwriting threshold."})
    else:
        factors.append({"factor": "Debt-to-Income (DTI)", "impact": "Negative", "detail": f"Elevated DTI of {dti_ratio:.1f}%; debt repayment strains monthly income."})

    if total_income >= 5000:
        factors.append({"factor": "Household Total Income", "impact": "Positive", "detail": f"Strong combined income of ${total_income:,.0f}."})
    else:
        factors.append({"factor": "Household Total Income", "impact": "Neutral", "detail": f"Combined income of ${total_income:,.0f} supports lower loan bands."})

    return {
        "status": status,
        "is_approved": pred == 1,
        "confidence": confidence,
        "approval_probability": approval_prob,
        "rejection_probability": round(prob_rejected * 100, 1),
        "risk_level": risk_level,
        "risk_color": risk_color,
        "monthly_emi": round(emi, 2),
        "total_income": total_income,
        "balance_income": round(balance_income, 2),
        "dti_ratio": round(dti_ratio, 1),
        "factors": factors
    }

@app.route('/')
def home():
    """Landing and executive overview page."""
    stats = {
        'total_applicants': 614,
        'overall_approval_rate': "68.7%",
        'best_model': preprocessing.get('best_model', 'Random Forest') if preprocessing else 'Random Forest',
        'top_accuracy': "86.2%",
        'features_count': len(preprocessing.get('feature_cols', [])) if preprocessing else 14
    }
    return render_template('index.html', stats=stats)

@app.route('/predict', methods=['GET', 'POST'])
def predict_page():
    """Loan prediction form & real-time assessment."""
    result = None
    form_data = {}

    if request.method == 'POST':
        form_data = request.form.to_dict()
        result = compute_prediction(form_data)

    return render_template('prediction.html', result=result, form_data=form_data)

@app.route('/dashboard')
def dashboard():
    """Analytics, Model Leaderboard, and EDA Hub."""
    return render_template(
        'dashboard.html',
        metrics=metrics_data,
        feature_importance=feature_importance_data,
        best_model=preprocessing.get('best_model', 'Random Forest') if preprocessing else 'Random Forest'
    )

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """REST API endpoint for loan approval prediction."""
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No input JSON provided"}), 400
        result = compute_prediction(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/metrics', methods=['GET'])
def api_metrics():
    """API endpoint to fetch model evaluation metrics."""
    return jsonify(metrics_data)

@app.route('/api/feature-importance', methods=['GET'])
def api_feature_importance():
    """API endpoint to fetch feature importance ranking."""
    return jsonify(feature_importance_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
