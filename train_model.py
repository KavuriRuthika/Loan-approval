"""
=============================================================================
AI-Based Loan Approval Prediction and Financial Risk Analysis System
Model Training, Evaluation, and Serialization Pipeline
=============================================================================
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

# Setup directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "loan_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
IMG_DIR = os.path.join(BASE_DIR, "static", "images")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

# Styling for Matplotlib & Seaborn
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#e2e8f0'
plt.rcParams['axes.linewidth'] = 1.2

def generate_eda_visualizations(df_raw):
    """Generate and save publication-quality EDA charts."""
    print("Generating EDA visualizations...")

    # 1. Loan Approval Distribution
    fig, ax = plt.subplots(figsize=(6, 5))
    palette = {'Y': '#10b981', 'N': '#ef4444'}
    counts = df_raw['Loan_Status'].value_counts()
    bars = ax.bar(['Approved (Y)', 'Rejected (N)'], [counts.get('Y', 0), counts.get('N', 0)], color=['#10b981', '#ef4444'], width=0.5)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 5, f"{yval} ({yval/len(df_raw)*100:.1f}%)", ha='center', va='bottom', fontweight='bold', color='#1e293b')
    ax.set_title("Overall Loan Approval Distribution", fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel("Number of Applicants", fontsize=11)
    ax.set_ylim(0, max(counts) + 50)
    plt.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "eda_loan_status.png"), dpi=200)
    plt.close(fig)

    # 2. Credit History vs Loan Approval
    fig, ax = plt.subplots(figsize=(7, 5))
    ch_df = df_raw.dropna(subset=['Credit_History'])
    ct = pd.crosstab(ch_df['Credit_History'], ch_df['Loan_Status'], normalize='index') * 100
    ct.plot(kind='bar', stacked=True, color=['#ef4444', '#10b981'], ax=ax, width=0.45)
    ax.set_title("Loan Approval by Credit History Record", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Credit History (0.0 = Poor / Default, 1.0 = Good / Repaid)", fontsize=11)
    ax.set_ylabel("Approval Percentage (%)", fontsize=11)
    ax.set_xticklabels(["Bad Credit (0.0)", "Good Credit (1.0)"], rotation=0)
    ax.legend(["Rejected (N)", "Approved (Y)"], loc='upper left')
    plt.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "eda_credit_history.png"), dpi=200)
    plt.close(fig)

    # 3. Property Area vs Loan Approval
    fig, ax = plt.subplots(figsize=(7, 5))
    pa_ct = pd.crosstab(df_raw['Property_Area'], df_raw['Loan_Status'], normalize='index') * 100
    pa_ct.plot(kind='bar', color=['#ef4444', '#10b981'], ax=ax, width=0.5)
    ax.set_title("Approval Rates Across Property Geographic Sectors", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Property Area", fontsize=11)
    ax.set_ylabel("Percentage (%)", fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(["Rejected (N)", "Approved (Y)"], loc='upper right')
    plt.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "eda_property_area.png"), dpi=200)
    plt.close(fig)

    # 4. Education vs Loan Approval
    fig, ax = plt.subplots(figsize=(7, 5))
    edu_ct = pd.crosstab(df_raw['Education'], df_raw['Loan_Status'], normalize='index') * 100
    edu_ct.plot(kind='bar', color=['#ef4444', '#10b981'], ax=ax, width=0.45)
    ax.set_title("Loan Approval by Education Level", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Applicant Education", fontsize=11)
    ax.set_ylabel("Percentage (%)", fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(["Rejected (N)", "Approved (Y)"], loc='upper right')
    plt.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "eda_education.png"), dpi=200)
    plt.close(fig)

    # 5. Income Distributions & Boxplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(df_raw['ApplicantIncome'], kde=True, ax=ax1, color='#3b82f6', bins=30)
    ax1.set_title("Applicant Income Distribution (Skewed)", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Applicant Income ($)")

    sns.boxplot(x='Loan_Status', y='LoanAmount', data=df_raw, ax=ax2, palette={'Y': '#10b981', 'N': '#ef4444'})
    ax2.set_title("Requested Loan Amount by Approval Status", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Loan Status")
    ax2.set_ylabel("Loan Amount ($ in thousands)")
    plt.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "eda_income_dist.png"), dpi=200)
    plt.close(fig)

    # 6. Correlation Heatmap (for numerical features)
    fig, ax = plt.subplots(figsize=(8, 6))
    num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
    corr_df = df_raw[num_cols].dropna().corr()
    sns.heatmap(corr_df, annot=True, cmap='Blues', fmt='.2f', linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_title("Numerical Attributes Correlation Matrix", fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "eda_correlation.png"), dpi=200)
    plt.close(fig)


def train_and_evaluate():
    print(f"Loading dataset from: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)

    # Generate EDA graphics
    generate_eda_visualizations(df)

    # 1. Missing Value Imputation Specs
    imputation_values = {
        'Gender': df['Gender'].mode()[0],
        'Married': df['Married'].mode()[0],
        'Dependents': df['Dependents'].mode()[0],
        'Self_Employed': df['Self_Employed'].mode()[0],
        'LoanAmount': float(df['LoanAmount'].median()),
        'Loan_Amount_Term': float(df['Loan_Amount_Term'].mode()[0]),
        'Credit_History': float(df['Credit_History'].mode()[0]),
    }

    df_clean = df.copy()
    if 'Loan_ID' in df_clean.columns:
        df_clean = df_clean.drop('Loan_ID', axis=1)

    for col, val in imputation_values.items():
        df_clean[col] = df_clean[col].fillna(val)

    # 2. Feature Engineering
    df_clean['TotalIncome'] = df_clean['ApplicantIncome'] + df_clean['CoapplicantIncome']
    # EMI calculation: LoanAmount is in thousands, term is in months
    df_clean['EMI'] = (df_clean['LoanAmount'] * 1000) / df_clean['Loan_Amount_Term']
    df_clean['BalanceIncome'] = df_clean['TotalIncome'] - df_clean['EMI']

    # 3. Categorical Encoders
    cat_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
    encoders = {}
    df_encoded = df_clean.copy()

    for col in cat_columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        encoders[col] = {
            'classes': [str(c) for c in le.classes_],
            'mapping': {str(cls_): int(idx) for idx, cls_ in enumerate(le.classes_)}
        }

    # Features and Target
    feature_cols = [
        'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
        'Credit_History', 'Property_Area', 'TotalIncome', 'EMI', 'BalanceIncome'
    ]

    X = df_encoded[feature_cols]
    y = df_encoded['Loan_Status']

    # Train/Test Split (80% train, 20% test stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Scale numerical features for models that require scaling (like Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Candidate Algorithms
    models = {
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=1000, random_state=42, C=1.0),
            'use_scaled': True
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=42),
            'use_scaled': False
        },
        'Random Forest': {
            'model': RandomForestClassifier(n_estimators=250, max_depth=6, min_samples_leaf=3, random_state=42),
            'use_scaled': False
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42),
            'use_scaled': False
        }
    }

    metrics_results = {}
    best_model_name = 'Random Forest'

    print("\n" + "="*70)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*70)

    for name, config in models.items():
        clf = config['model']
        x_tr = X_train_scaled if config['use_scaled'] else X_train
        x_te = X_test_scaled if config['use_scaled'] else X_test

        clf.fit(x_tr, y_train)
        y_pred = clf.predict(x_te)
        y_prob = clf.predict_proba(x_te)[:, 1] if hasattr(clf, 'predict_proba') else y_pred

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        # Stratified 5-Fold Cross Validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(clf, X_train_scaled if config['use_scaled'] else X_train, y_train, cv=cv, scoring='accuracy')

        cm = confusion_matrix(y_test, y_pred).tolist()
        report = classification_report(y_test, y_pred, output_dict=True)

        metrics_results[name] = {
            'accuracy': round(float(acc), 4),
            'precision': round(float(prec), 4),
            'recall': round(float(rec), 4),
            'f1_score': round(float(f1), 4),
            'roc_auc': round(float(auc), 4),
            'cv_mean': round(float(cv_scores.mean()), 4),
            'cv_std': round(float(cv_scores.std()), 4),
            'confusion_matrix': cm,
            'classification_report': report
        }

        print(f"{name:20s} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f} | 5-Fold CV: {cv_scores.mean():.4f}")

    # Plot Model Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(10, 5.5))
    model_names = list(metrics_results.keys())
    accuracies = [metrics_results[m]['accuracy'] * 100 for m in model_names]
    f1_scores = [metrics_results[m]['f1_score'] * 100 for m in model_names]
    roc_aucs = [metrics_results[m]['roc_auc'] * 100 for m in model_names]

    x = np.arange(len(model_names))
    width = 0.25

    rects1 = ax.bar(x - width, accuracies, width, label='Accuracy (%)', color='#3b82f6')
    rects2 = ax.bar(x, f1_scores, width, label='F1-Score (%)', color='#10b981')
    rects3 = ax.bar(x + width, roc_aucs, width, label='ROC-AUC (%)', color='#8b5cf6')

    ax.set_ylabel('Score (%)', fontsize=11)
    ax.set_title('Comparative Benchmark of Machine Learning Algorithms', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=10, fontweight='bold')
    ax.set_ylim(60, 100)
    ax.legend(loc='lower right')

    for rects in [rects1, rects2, rects3]:
        for bar in rects:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "eda_model_comparison.png"), dpi=200)
    plt.close(fig)

    # Feature Importance (Random Forest)
    rf_model = models['Random Forest']['model']
    importances = rf_model.feature_importances_
    feat_importance_dict = {feat: float(imp) for feat, imp in zip(feature_cols, importances)}
    sorted_feat = sorted(feat_importance_dict.items(), key=lambda x: x[1], reverse=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    feats = [item[0] for item in sorted_feat][::-1]
    vals = [item[1] * 100 for item in sorted_feat][::-1]
    ax.barh(feats, vals, color='#0284c7', height=0.6)
    ax.set_title("Random Forest Feature Importance Analysis", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Relative Importance (%)", fontsize=11)
    for i, v in enumerate(vals):
        ax.text(v + 0.5, i, f"{v:.2f}%", va='center', fontweight='bold', fontsize=9, color='#1e293b')
    plt.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "eda_feature_importance.png"), dpi=200)
    plt.close(fig)

    # Save Selected Best Model & Preprocessing Pipeline
    best_clf = models[best_model_name]['model']
    model_save_path = os.path.join(MODEL_DIR, "loan_model.pkl")
    scaler_save_path = os.path.join(MODEL_DIR, "scaler.pkl")
    prep_save_path = os.path.join(MODEL_DIR, "preprocessing.pkl")

    joblib.dump(best_clf, model_save_path)
    joblib.dump(scaler, scaler_save_path)

    preprocessing_bundle = {
        'imputation_values': imputation_values,
        'encoders': encoders,
        'feature_cols': feature_cols,
        'cat_columns': cat_columns,
        'best_model': best_model_name
    }
    joblib.dump(preprocessing_bundle, prep_save_path)

    # Save Metrics & Feature Importance to JSON
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_results, f, indent=4)

    with open(os.path.join(MODEL_DIR, "feature_importance.json"), "w", encoding="utf-8") as f:
        json.dump(dict(sorted_feat), f, indent=4)

    print("\n" + "="*70)
    print("SUCCESSFULLY COMPLETED TRAINING & SERIALIZATION!")
    print(f"Model saved to: {model_save_path}")
    print(f"Preprocessing bundle saved to: {prep_save_path}")
    print(f"Metrics & charts stored in model/ and static/images/")
    print("="*70)

if __name__ == '__main__':
    train_and_evaluate()
