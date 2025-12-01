import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

st.title("📉 Customer Churn Prediction App (with SHAP & Visualizations)")

uploaded_file = st.file_uploader("Upload Telco Customer Churn Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    if "customerID" in df.columns:
        df = df.drop(["customerID"], axis=1)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce').fillna(df["TotalCharges"].median())

    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col])

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    st.success("Model Trained Successfully!")

    st.subheader("📊 Churn Influencing Factors (Feature Importance)")
    fig, ax = plt.subplots(figsize=(8,5))
    feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    sns.barplot(x=feat_imp.values, y=feat_imp.index)
    st.pyplot(fig)

    st.subheader("🔍 Test a Customer for Churn Prediction")
    sample = X_test.iloc[0:1]
    pred = model.predict(sample)[0]

    st.write("Prediction:", "❌ Churn" if pred == 1 else "✅ Not Churn")

    st.subheader("🧠 SHAP Explanation (Why this prediction?)")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    st.write("### SHAP Summary Plot")
    fig2 = plt.figure()
    shap.summary_plot(shap_values[1], X_test, show=False)
    st.pyplot(fig2)

    st.write("### SHAP Force Plot (Single Customer)")
    sample_idx = 0
    force_html = shap.force_plot(explainer.expected_value[1], shap_values[1][sample_idx], X_test.iloc[sample_idx], matplotlib=False)
    st.components.v1.html(force_html.html(), height=300)
