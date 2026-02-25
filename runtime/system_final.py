import pandas as pd
import joblib

from src.dt_llm.digital_twin.dt_predictor_v2 import DigitalTwinV2
from src.dt_llm.decision.anomaly_gate_v2 import AnomalyGate

# ----------------------------
# Configuration
# ----------------------------

DATA_PATH = "data/raw/month_10percent_faults.xlsx"
DT_MODEL_PATH = "data/processed/dt_v2_ridge.joblib"
CLF_MODEL_PATH = "models/behavior_classifier_v2.joblib"

COLDT = "Date Time"
T, Td, RH = "T (degC)", "Tdew (degC)", "rh (%)"


# ----------------------------
# Feature Builder (Behavior)
# ----------------------------

def build_features(df, i):
    if i == 0:
        return None

    dT = df[T].iloc[i] - df[T].iloc[i - 1]
    dTd = df[Td].iloc[i] - df[Td].iloc[i - 1]
    dRH = df[RH].iloc[i] - df[RH].iloc[i - 1]

    rolling_T_std = df[T].iloc[max(0, i - 5): i + 1].std()
    rolling_RH_std = df[RH].iloc[max(0, i - 5): i + 1].std()

    return [
        df[T].iloc[i],
        df[Td].iloc[i],
        df[RH].iloc[i],
        dT,
        dTd,
        dRH,
        rolling_T_std if pd.notna(rolling_T_std) else 0,
        rolling_RH_std if pd.notna(rolling_RH_std) else 0,
    ]


# ----------------------------
# LLM 
# ----------------------------

def send_to_llm(sensor_row, prediction, anomaly_label):
    # Placeholder for multi-agent integration
    pass


# ----------------------------
# Main System Loop
# ----------------------------

def main():

    # Load models
    dt = DigitalTwinV2(DT_MODEL_PATH)
    behavior_clf = joblib.load(CLF_MODEL_PATH)
    gate = AnomalyGate()

    # Load data
    df = pd.read_excel(DATA_PATH)
    df[COLDT] = pd.to_datetime(df[COLDT], format="%d.%m.%Y %H:%M:%S")

    # Initialize DT
    first = df.iloc[0]
    dt.update_state(first[T], first[Td], first[RH])

    for i in range(1, len(df)):

        row = df.iloc[i]

        # Sensor → DT
        prediction = dt.predict(row[COLDT])

        #  Sensor → Classifier
        features = build_features(df, i)
        prob = behavior_clf.predict_proba([features])[0][1]
        anomaly_label = int(prob > 0.5)

        #  Send everything to LLM
        send_to_llm(row, prediction, anomaly_label)

        #  Classifier → DT (via gate)
        gate.update_state(dt, row, prediction, anomaly_label)

    print("System Final run completed.")


if __name__ == "__main__":
    main()
