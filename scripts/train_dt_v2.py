import glob
import os
import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Column Names
COLDT = "Date Time"
T = "T (degC)"
Td = "Tdew (degC)"
RH = "rh (%)"


def build_features(df: pd.DataFrame) -> np.ndarray:
    hour = df[COLDT].dt.hour + df[COLDT].dt.minute / 60.0
    hour_ang = 2.0 * np.pi * hour / 24.0
    hour_sin = np.sin(hour_ang)
    hour_cos = np.cos(hour_ang)

    doy = df[COLDT].dt.dayofyear
    doy_ang = 2.0 * np.pi * doy / 365.0
    doy_sin = np.sin(doy_ang)
    doy_cos = np.cos(doy_ang)

    hum_x_hour_sin = df[RH].values * hour_sin
    hum_x_hour_cos = df[RH].values * hour_cos

    X = np.column_stack([
        df[T].values,
        df[Td].values,
        df[RH].values,
        hour_sin,
        hour_cos,
        doy_sin,
        doy_cos,
        hum_x_hour_sin,
        hum_x_hour_cos
    ])

    return X


def main():
    paths = sorted(glob.glob("data/raw/clean_months/*.xlsx"))

    if not paths:
        raise FileNotFoundError(
            "No files found in data/raw/clean_months/*.xlsx"
        )

    print("Files used for DT training:")
    for p in paths:
        print("  ", p)

    frames = []

    for p in paths:
        df = pd.read_excel(p)
        df[COLDT] = pd.to_datetime(
            df[COLDT],
            format="%d.%m.%Y %H:%M:%S",
            errors="coerce"
        )
        df = df.dropna(subset=[COLDT, T, Td, RH]).reset_index(drop=True)
        frames.append(df[[COLDT, T, Td, RH]])

    df = (
        pd.concat(frames, ignore_index=True)
        .sort_values(COLDT)
        .reset_index(drop=True)
    )

    print("\nTotal training rows:", len(df))

    # One-step targets
    df["T_next"] = df[T].shift(-1)
    df["Td_next"] = df[Td].shift(-1)
    df = df.dropna().reset_index(drop=True)

    X = build_features(df)
    yT = df["T_next"].values
    yTd = df["Td_next"].values

    split_idx = int(0.8 * len(df))

    X_train, X_test = X[:split_idx], X[split_idx:]
    yT_train, yT_test = yT[:split_idx], yT[split_idx:]
    yTd_train, yTd_test = yTd[:split_idx], yTd[split_idx:]

    T_model = Ridge(alpha=1.0)
    Td_model = Ridge(alpha=1.0)

    T_model.fit(X_train, yT_train)
    Td_model.fit(X_train, yTd_train)

    T_pred = T_model.predict(X_test)
    Td_pred = Td_model.predict(X_test)

    print("\nDT v2 Temporal Validation Metrics")
    print("T   MAE:", mean_absolute_error(yT_test, T_pred),
          "RMSE:", np.sqrt(mean_squared_error(yT_test, T_pred)))

    print("Td  MAE:", mean_absolute_error(yTd_test, Td_pred),
          "RMSE:", np.sqrt(mean_squared_error(yTd_test, Td_pred)))

    os.makedirs("data/processed", exist_ok=True)
    joblib.dump(
        {"T_model": T_model, "Td_model": Td_model},
        "data/processed/dt_v2_ridge.joblib"
    )

    print("\nModel saved to data/processed/dt_v2_ridge.joblib")


if __name__ == "__main__":
    main()
