#!/usr/bin/env python3
"""
Model Training & Comparison
Temporal Cross-Validation (2010-2021 train, 2022-2025 test)
"""

import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier


FEATURES = [
    "MaxTemp", "MinTemp", "AvgTemp", "Rainfall", "WindSpeed",
    "WarmDays_3", "ColdNights_3",
    "WarmDays_5", "ColdNights_5",
    "AvgMax_3", "AvgMin_3",
    "TempDrop", "WarmAfterCold", "StrongTrigger"
]


def train_models(X_train, X_test, y_train, y_test):
    """
    Train and evaluate all models
    """
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                class_weight="balanced",
                max_iter=1000
            ))
        ]),

        "Decision Tree": DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced"
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42
        ),
        
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    }

    results = []
    trained_models = {}

    for name, clf in models.items():
        print(f"\n{'='*50}")
        print(f"Training: {name}")
        print(f"{'='*50}")

        start_time = time.time()
        clf.fit(X_train, y_train)
        end_time = time.time()

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)

        print(f"Accuracy : {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall   : {recall:.3f}")
        print(f"F1 Score : {f1:.3f}")
        print(f"ROC-AUC  : {roc_auc:.3f}")
        print(f"Time     : {end_time - start_time:.2f} seconds")

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "ROC-AUC": roc_auc,
            "Time_seconds": end_time - start_time
        })

        trained_models[name] = clf

    return pd.DataFrame(results), trained_models


if __name__ == "__main__":
    print("[INFO] Loading data...")
    df = pd.read_csv("data/hagerstown_weather_2010_2025_with_features.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    
    # Temporal split: 2010-2021 train, 2022-2025 test
    train_df = df[df["Year"] <= 2021]
    test_df = df[df["Year"] > 2021]
    
    X_train = train_df[FEATURES]
    y_train = train_df["FlightObserved"]
    X_test = test_df[FEATURES]
    y_test = test_df["FlightObserved"]
    
    print(f"[INFO] Training samples: {len(X_train)}")
    print(f"[INFO] Testing samples: {len(X_test)}")
    print(f"[INFO] Positive training samples: {y_train.sum()}")
    print(f"[INFO] Positive testing samples: {y_test.sum()}")
    
    results_df, models = train_models(X_train, X_test, y_train, y_test)
    
    # Save results
    results_df.to_csv("results/model_comparison_results.csv", index=False)
    print("\n[SUCCESS] Results saved to results/model_comparison_results.csv")
    print(f"\n{results_df.to_string()}")
