"""
Run this ONCE, locally, before deploying.
It trains both models and saves them as small .pkl files, so the
Streamlit app never has to touch pandas/sklearn training code at runtime —
it just loads the finished model. This is what makes the app start fast.

Usage:
    python train_model.py
"""
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              roc_auc_score, r2_score, mean_absolute_error)

df = pd.read_csv("Telco-Customer-Churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)
df["Churn_bin"] = (df["Churn"] == "Yes").astype(int)

# ---- Model 1: Logistic Regression (churn) ----
log_features = ["tenure", "MonthlyCharges", "Contract", "InternetService",
                 "PaymentMethod", "PaperlessBilling", "SeniorCitizen", "Partner", "Dependents"]
X_log = df[log_features].copy()
cat_cols = ["Contract", "InternetService", "PaymentMethod", "PaperlessBilling", "Partner", "Dependents"]
X_log = pd.get_dummies(X_log, columns=cat_cols, drop_first=True)
y_log = df["Churn_bin"]

Xtr, Xte, ytr, yte = train_test_split(X_log, y_log, test_size=0.2, random_state=42, stratify=y_log)

scaler = StandardScaler()
num_cols = ["tenure", "MonthlyCharges", "SeniorCitizen"]
Xtr_s, Xte_s = Xtr.copy(), Xte.copy()
Xtr_s[num_cols] = scaler.fit_transform(Xtr[num_cols])
Xte_s[num_cols] = scaler.transform(Xte[num_cols])

logit = LogisticRegression(max_iter=2000, class_weight="balanced")
logit.fit(Xtr_s, ytr)
yp, ypr = logit.predict(Xte_s), logit.predict_proba(Xte_s)[:, 1]
logit_metrics = {
    "accuracy": accuracy_score(yte, yp), "precision": precision_score(yte, yp),
    "recall": recall_score(yte, yp), "roc_auc": roc_auc_score(yte, ypr),
}

# ---- Model 2: Linear Regression (customer value) ----
lin_features = ["tenure", "MonthlyCharges", "Contract", "InternetService", "PaymentMethod"]
X_lin = df[lin_features].copy()
X_lin = pd.get_dummies(X_lin, columns=["Contract", "InternetService", "PaymentMethod"], drop_first=True)
y_lin = df["TotalCharges"]
Xltr, Xlte, yltr, ylte = train_test_split(X_lin, y_lin, test_size=0.2, random_state=42)
linreg = LinearRegression()
linreg.fit(Xltr, yltr)
ylp = linreg.predict(Xlte)
lin_metrics = {"r2": r2_score(ylte, ylp), "mae": mean_absolute_error(ylte, ylp)}

# ---- Save everything the app needs, nothing more ----
bundle = {
    "logit": logit, "logit_cols": list(X_log.columns), "scaler": scaler, "num_cols": num_cols,
    "logit_metrics": logit_metrics, "churn_rate": float(y_log.mean()),
    "linreg": linreg, "lin_cols": list(X_lin.columns), "lin_metrics": lin_metrics,
}
with open("model_bundle.pkl", "wb") as f:
    pickle.dump(bundle, f)

print("Saved model_bundle.pkl")
print("Logistic:", logit_metrics)
print("Linear:", lin_metrics)
