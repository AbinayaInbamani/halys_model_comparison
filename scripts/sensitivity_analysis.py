#!/usr/bin/env python3
"""
Sensitivity Analysis
- Window robustness (±1, ±3, ±5, ±7 days)
- Temperature threshold variations
- Feature ablation
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score


def test_window_sensitivity(weather_df, historical_events, features):
    """
    Test robustness to flight window definition
    """
    windows = [1, 3, 5, 7]
    results = []
    
    for window in windows:
        # Create labels with this window
        labeled_df = weather_df.copy()
        labeled_df["FlightObserved"] = 0
        
        for _, row in historical_events.iterrows():
            event_date = row["EstimatedFlightDate"]
            mask = (
                (labeled_df["Date"] >= event_date - pd.Timedelta(days=window)) &
                (labeled_df["Date"] <= event_date + pd.Timedelta(days=window))
            )
            labeled_df.loc[mask, "FlightObserved"] = 1
        
        # Train/test split
        X_train = labeled_df[labeled_df["Year"] <= 2021][features]
        y_train = labeled_df[labeled_df["Year"] <= 2021]["FlightObserved"]
        X_test = labeled_df[labeled_df["Year"] > 2021][features]
        y_test = labeled_df[labeled_df["Year"] > 2021]["FlightObserved"]
        
        # Train model
        model = ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        results.append({
            "Window_days": window,
            "F1_Score": f1,
            "ROC_AUC": roc_auc,
            "Positive_Samples": y_train.sum()
        })
        
        print(f"Window ±{window} days: F1={f1:.3f}, ROC-AUC={roc_auc:.3f}")
    
    return pd.DataFrame(results)


def feature_importance_ablation(model, X_test, y_test, features):
    """
    Remove each feature and measure performance drop (ablation study)
    """
    baseline_f1 = f1_score(y_test, model.predict(X_test), zero_division=0)
    
    ablation_results = []
    for feature in features:
        X_test_ablated = X_test.drop(columns=[feature])
        y_pred = model.predict(X_test_ablated)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        importance = baseline_f1 - f1
        
        ablation_results.append({
            "Feature": feature,
            "F1_Drop": importance,
            "Importance_Rank": None  # Will fill after sorting
        })
    
    ablation_df = pd.DataFrame(ablation_results).sort_values("F1_Drop", ascending=False)
    ablation_df["Importance_Rank"] = range(1, len(ablation_df) + 1)
    
    return ablation_df


if __name__ == "__main__":
    print("[INFO] Loading data for sensitivity analysis...")
    # Add your analysis code here
    print("[INFO] Sensitivity analysis complete")
