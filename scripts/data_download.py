#!/usr/bin/env python3
"""
NASA POWER Weather Data Download
For BMSB Phenology Prediction
"""

import pandas as pd
import requests
import sys
from datetime import datetime

HAGERSTOWN_LAT = 39.6418
HAGERSTOWN_LON = -77.7200

def get_nasa_power_weather(lat, lon, start_date, end_date):
    """
    Download daily weather data from NASA POWER API
    
    Parameters:
    -----------
    lat, lon : float
        Location coordinates
    start_date, end_date : str
        'YYYY-MM-DD' format
    
    Returns:
    --------
    pd.DataFrame with columns:
        Date, MaxTemp (°F), MinTemp (°F), Rainfall (mm), WindSpeed (m/s), Year
    """
    start = pd.to_datetime(start_date).strftime("%Y%m%d")
    end = pd.to_datetime(end_date).strftime("%Y%m%d")

    url = "https://power.larc.nasa.gov/api/temporal/daily/point"

    params = {
        "parameters": "T2M_MAX,T2M_MIN,PRECTOTCORR,WS2M",
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": start,
        "end": end,
        "format": "JSON"
    }

    print(f"[INFO] Downloading weather data from {start_date} to {end_date}...")
    response = requests.get(url, params=params)
    response.raise_for_status()

    data = response.json()["properties"]["parameter"]

    df = pd.DataFrame({
        "Date": pd.to_datetime(list(data["T2M_MAX"].keys())),
        "MaxTemp": list(data["T2M_MAX"].values()),
        "MinTemp": list(data["T2M_MIN"].values()),
        "Rainfall": list(data["PRECTOTCORR"].values()),
        "WindSpeed": list(data["WS2M"].values())
    })

    # Convert Celsius to Fahrenheit
    df["MaxTemp"] = df["MaxTemp"] * 9 / 5 + 32
    df["MinTemp"] = df["MinTemp"] * 9 / 5 + 32
    df["Year"] = df["Date"].dt.year

    print(f"[SUCCESS] Downloaded {len(df)} weather records")
    return df


if __name__ == "__main__":
    weather = get_nasa_power_weather(
        HAGERSTOWN_LAT,
        HAGERSTOWN_LON,
        "2010-09-01",
        "2025-10-31"
    )
    
    # Filter Sep-Oct only
    weather = weather[weather["Date"].dt.month.isin([9, 10])]
    
    # Save
    weather.to_csv("data/hagerstown_weather_2010_2025.csv", index=False)
    print(f"[SUCCESS] Saved {len(weather)} Sep-Oct records to data/hagerstown_weather_2010_2025.csv")
