import pandas as pd

# =========================
# =========================
df = pd.read_csv("churn.csv")

print("\nInitial Data Info:\n")
print(df.info())

# =========================
# STEP 2: Data Cleaning
# =========================

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Fill missing values
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# Drop customerID (not useful)
df.drop("customerID", axis=1, inplace=True)

# Convert categorical to numeric
df = pd.get_dummies(df)

print("\nAfter Cleaning:")
print(df.shape)

# =========================
# STEP 3: Train Model
# =========================
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X = df[["tenure", "MonthlyCharges"]]
y = df["Churn_Yes"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

# =========================
# STEP 4: SHAP Explanation
# =========================
import shap
import matplotlib.pyplot as plt

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Plot
plt.figure(figsize=(12,7))
shap.summary_plot(shap_values, X_test, plot_type="bar", max_display=10)
plt.show()

# =========================
# STEP 5: Recommendation System
# =========================
def give_recommendation(row):
    if row['tenure'] < 12:
        return "Offer loyalty discount"
    elif row['MonthlyCharges'] > 80:
        return "Suggest cheaper plan"
    else:
        return "Provide customer support follow-up"

# Take sample
sample = X_test.copy()
sample["Churn_Pred"] = y_pred

# Add recommendations
sample["Recommendation"] = sample.apply(give_recommendation, axis=1)

# =========================
# FINAL OUTPUT
# =========================
print("\n========== FINAL OUTPUT ==========\n")
print(sample[['Churn_Pred', 'Recommendation']].head(10).to_string(index=False))
import joblib
joblib.dump(model, "model.pkl")
joblib.dump(X.columns.tolist(), "columns.pkl")