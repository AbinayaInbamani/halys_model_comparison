#!/usr/bin/env python3
"""
Feature Engineering for BMSB Flight Prediction
Ecologically-grounded temperature and wind features
"""

import pandas as pd
import numpy as np

def add_features(df):
    """
    Add engineered features based on BMSB ecological requirements
    
    Temperature thresholds (Celsius → Fahrenheit):
    - 50°F (10°C): BMSB flight threshold
    - 70°F (21°C): Optimal flight temperature
    - 75-85°F (24-29°C): Peak flight activity
    
    Parameters:
    -----------
    df : pd.DataFrame
        Must contain: Date, MaxTemp, MinTemp, Rainfall, WindSpeed, Year
    
    Returns:
    --------
    pd.DataFrame with added features
    """
    df = df.copy()
    df = df.sort_values(["Year", "Date"])

    # Basic temperature feature
    df["AvgTemp"] = (df["MaxTemp"] + df["MinTemp"]) / 2

    # 3-day rolling window features
    df["WarmDays_3"] = df.groupby("Year")["MaxTemp"].transform(
        lambda x: (x > 70).rolling(3, min_periods=1).sum()
    )

    df["ColdNights_3"] = df.groupby("Year")["MinTemp"].transform(
        lambda x: (x < 50).rolling(3, min_periods=1).sum()
    )

    # 5-day rolling window features
    df["WarmDays_5"] = df.groupby("Year")["MaxTemp"].transform(
        lambda x: (x > 70).rolling(5, min_periods=1).sum()
    )

    df["ColdNights_5"] = df.groupby("Year")["MinTemp"].transform(
        lambda x: (x < 50).rolling(5, min_periods=1).sum()
    )

    # Rolling means
    df["AvgMax_3"] = df.groupby("Year")["MaxTemp"].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )

    df["AvgMin_3"] = df.groupby("Year")["MinTemp"].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )

    # Temperature dynamics
    df["TempDrop"] = df.groupby("Year")["MaxTemp"].diff()

    # Binary ecological triggers
    df["WarmAfterCold"] = np.where(
        (df["MinTemp"] <= 50) & (df["MaxTemp"] >= 65), 1, 0
    )

    df["StrongTrigger"] = np.where(
        (df["MinTemp"] <= 50) &
        (df["MaxTemp"] >= 70) &
        (df["Rainfall"] <= 0.1) &
        (df["WindSpeed"] <= 8),
        1,
        0
    )

    return df.fillna(0)


if __name__ == "__main__":
    # Example usage
    df = pd.read_csv("data/hagerstown_weather_2010_2025.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    
    df_features = add_features(df)
    
    print("[SUCCESS] Features added")
    print(f"Feature columns: {list(df_features.columns)}")
    print(f"\nShape: {df_features.shape}")
