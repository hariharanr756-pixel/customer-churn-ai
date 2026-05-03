import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# -------------------------
# PAGE
# -------------------------
st.set_page_config(page_title="Churn Predictor", layout="centered")

st.markdown("<h1 style='text-align:center;'>📊 Customer Churn Predictor</h1>", unsafe_allow_html=True)
st.caption("Simple • Clean • Stable")

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv("churn.csv")

# Clean data
if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)

df.replace(" ", np.nan, inplace=True)

if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

df = df.dropna()

# Target
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

X = df.drop("Churn", axis=1)
y = df["Churn"]

# -------------------------
# MODEL PIPELINE (NO ERRORS)
# -------------------------
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(exclude=["int64", "float64"]).columns

model = Pipeline([
    ("prep", ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ])),
    ("rf", RandomForestClassifier(n_estimators=100, random_state=42))
])

model.fit(X, y)

# -------------------------
# CLEAN INPUT UI (ONLY IMPORTANT)
# -------------------------
st.subheader("Enter Customer Details")

tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly = st.number_input("Monthly Charges", value=50.0)

contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer", "Credit card"
])

senior = st.radio("Senior Citizen", ["No", "Yes"])

# -------------------------
# BUILD INPUT (AUTO MATCH)
# -------------------------
input_df = pd.DataFrame([{
    "tenure": tenure,
    "MonthlyCharges": monthly,
    "Contract": contract,
    "InternetService": internet,
    "PaymentMethod": payment,
    "SeniorCitizen": 1 if senior == "Yes" else 0
}])

# Fill missing columns automatically
for col in X.columns:
    if col not in input_df.columns:
        if col in cat_cols:
            input_df[col] = df[col].mode()[0]
        else:
            input_df[col] = 0

input_df = input_df[X.columns]

# -------------------------
# PREDICTION
# -------------------------
if st.button("Predict"):

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.markdown("---")

    if pred == 1:
        st.error(f"⚠️ High Risk of Churn ({prob:.2f})")
    else:
        st.success(f"✅ Low Risk ({prob:.2f})")

    # -------------------------
    # CLEAN FEATURE GRAPH
    # -------------------------
    st.subheader("📊 Key Factors")

    rf = model.named_steps["rf"]
    feature_names = model.named_steps["prep"].get_feature_names_out()

    importances = rf.feature_importances_
    imp_series = pd.Series(importances, index=feature_names)

    # Clean labels
    clean_names = []
    for name in imp_series.index:
        name = name.replace("num__", "").replace("cat__", "")
        name = name.replace("_", " ")
        clean_names.append(name)

    imp_series.index = clean_names

    # Top features only
    top = imp_series.sort_values(ascending=False).head(6)
    top = top[::-1]

    # Plot
    fig, ax = plt.subplots()
    ax.barh(top.index, top.values)

    ax.set_title("Top Influencing Factors")
    ax.set_xlabel("Impact")

    st.pyplot(fig)

# -------------------------
# FOOTER
# -------------------------
st.markdown("---")
st.caption("🚀 Clean UI • Stable Model • Ready to Deploy")