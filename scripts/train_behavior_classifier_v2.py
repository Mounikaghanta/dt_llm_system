import glob
import os
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Columns
COLDT = "Date Time"
T, Td, RH = "T (degC)", "Tdew (degC)", "rh (%)"
LABEL = "binary_label"

def build_features(df):
    # Temporal features
    df["dT"] = df[T].diff().fillna(0)
    df["dTd"] = df[Td].diff().fillna(0)
    df["dRH"] = df[RH].diff().fillna(0)

    df["rolling_T_std"] = df[T].rolling(6).std().fillna(0)
    df["rolling_RH_std"] = df[RH].rolling(6).std().fillna(0)

    return df[[T, Td, RH, "dT", "dTd", "dRH", "rolling_T_std", "rolling_RH_std"]].values


def main():
    paths = sorted(glob.glob("data/raw/anomaly_months/*.xlsx"))
    if not paths:
        raise FileNotFoundError("No files found in data/raw/anomaly_months/*.xlsx")

    frames = []
    for p in paths:
        df = pd.read_excel(p)
        df[COLDT] = pd.to_datetime(df[COLDT], format="%d.%m.%Y %H:%M:%S", errors="coerce")
        df = df.dropna(subset=[COLDT, T, Td, RH, LABEL]).reset_index(drop=True)
        frames.append(df[[COLDT, T, Td, RH, LABEL]])

    df = pd.concat(frames, ignore_index=True).sort_values(COLDT).reset_index(drop=True)

    X = build_features(df)
    y = df[LABEL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    os.makedirs("outputs", exist_ok=True)
    joblib.dump(clf, "outputs/behavior_classifier_v2.joblib")
    print("\nSaved: outputs/behavior_classifier_v2.joblib")


if __name__ == "__main__":
    main()
