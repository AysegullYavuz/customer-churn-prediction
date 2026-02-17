import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix
)

from xgboost import XGBClassifier

# LOAD DATA

def load_data():
    return pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# PREPROCESSING

def preprocess_data(df):

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()
    df = df.drop("customerID", axis=1)

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df

# FEATURE PREPARATION

def prepare_features(df):

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X = pd.get_dummies(X, drop_first=True)

    return X, y

# MODEL TRAINING

def train_models(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # LOGISTIC REGRESSION

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )

    log_model.fit(X_train_scaled, y_train)

    y_prob_log = log_model.predict_proba(X_test_scaled)[:, 1]
    threshold = 0.4
    y_pred_log = (y_prob_log >= threshold).astype(int)

    print("\n==============================")
    print("LOGISTIC REGRESSION (Final)")
    print("==============================")
    print(f"Threshold: {threshold}")
    print(classification_report(y_test, y_pred_log))
    print("ROC AUC:", roc_auc_score(y_test, y_prob_log))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_log))

    # Feature Importance (Logistic)
    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": log_model.coef_[0]
    })

    importance_df["Abs_Coefficient"] = importance_df["Coefficient"].abs()
    importance_df = importance_df.sort_values("Abs_Coefficient", ascending=False)

    print("\nTop 10 Logistic Features")
    print(importance_df.head(10)[["Feature", "Coefficient"]])

    # Plot
    top10_log = importance_df.head(10)

    plt.figure(figsize=(10, 6))
    plt.barh(top10_log["Feature"], top10_log["Coefficient"])
    plt.xlabel("Coefficient Value")
    plt.title("Top 10 Logistic Regression Features")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("logistic_feature_importance.png", dpi=300)
    plt.show()

    # XGBOOST

    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

    xgb_model = XGBClassifier(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss"
    )

    xgb_model.fit(X_train, y_train)

    y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
    y_pred_xgb = (y_prob_xgb >= 0.5).astype(int)

    print("\n==============================")
    print("XGBOOST")
    print("==============================")
    print(classification_report(y_test, y_pred_xgb))
    print("ROC AUC:", roc_auc_score(y_test, y_prob_xgb))

    # Feature Importance (XGBoost)
    xgb_importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": xgb_model.feature_importances_
    }).sort_values("Importance", ascending=False)

    print("\nTop 10 XGBoost Features")
    print(xgb_importance.head(10))

    # Plot
    top10_xgb = xgb_importance.head(10)

    plt.figure(figsize=(10, 6))
    plt.barh(top10_xgb["Feature"], top10_xgb["Importance"])
    plt.xlabel("Importance Score")
    plt.title("Top 10 XGBoost Features")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("xgboost_feature_importance.png", dpi=300)
    plt.show()

# MAIN

def main():
    df = load_data()
    df = preprocess_data(df)
    X, y = prepare_features(df)
    train_models(X, y)


if __name__ == "__main__":
    main()
