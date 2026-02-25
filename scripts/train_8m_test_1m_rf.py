import glob
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Columns
COLDT = "Date Time"
T, Td, RH = "T (degC)", "Tdew (degC)", "rh (%)"
LABEL = "binary_label"

TRAIN_GLOB = "data/raw/anomaly_months/*.xlsx"
TEST_PATH = "data/raw/test_balanced_faults.xlsx"


def build_features(df):
    df = df.copy()

    df["dT"] = df[T].diff().fillna(0)
    df["dTd"] = df[Td].diff().fillna(0)
    df["dRH"] = df[RH].diff().fillna(0)

    df["rolling_T_std"] = df[T].rolling(6).std().fillna(0)
    df["rolling_RH_std"] = df[RH].rolling(6).std().fillna(0)

    return df[[T, Td, RH, "dT", "dTd", "dRH", "rolling_T_std", "rolling_RH_std"]].values


def load_data(paths):
    frames = []
    for p in paths:
        df = pd.read_excel(p)
        df[COLDT] = pd.to_datetime(df[COLDT], format="%d.%m.%Y %H:%M:%S", errors="coerce")
        df = df.dropna(subset=[COLDT, T, Td, RH, LABEL]).reset_index(drop=True)
        frames.append(df[[COLDT, T, Td, RH, LABEL]])

    return pd.concat(frames, ignore_index=True).sort_values(COLDT).reset_index(drop=True)


def main():
    # ---- TRAIN ----
    train_paths = sorted(glob.glob(TRAIN_GLOB))
    train_df = load_data(train_paths)

    X_train = build_features(train_df)
    y_train = train_df[LABEL].values

    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    clf.fit(X_train, y_train)

    # ---- TEST ----
    test_df = load_data([TEST_PATH])

    X_test = build_features(test_df)
    y_test = test_df[LABEL].values

    y_pred = clf.predict(X_test)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
