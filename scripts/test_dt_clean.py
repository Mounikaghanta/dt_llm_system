import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.dt_llm.digital_twin.dt_predictor_v2 import DigitalTwinV2

COLDT = "Date Time"
T, Td, RH = "T (degC)", "Tdew (degC)", "rh (%)"

def rmse(y, yhat):
    return np.sqrt(mean_squared_error(y, yhat))

dt = DigitalTwinV2("data/processed/dt_v2_ridge.joblib")

df = pd.read_excel("data/raw/month1.xlsx")
df[COLDT] = pd.to_datetime(df[COLDT], format="%d.%m.%Y %H:%M:%S")

first = df.iloc[0]
dt.update_state(first[T], first[Td], first[RH])

T_true, T_pred = [], []
Td_true, Td_pred = [], []
RH_true, RH_pred = [], []

for i in range(1, len(df)):
    row = df.iloc[i]

    pred = dt.predict(row[COLDT])

    T_true.append(row[T])
    Td_true.append(row[Td])
    RH_true.append(row[RH])

    T_pred.append(pred["T_pred"])
    Td_pred.append(pred["Td_pred"])
    RH_pred.append(pred["RH_pred"])

    dt.update_state(row[T], row[Td], row[RH])

print("DT Clean Evaluation")
print("T MAE:", mean_absolute_error(T_true, T_pred),
      "RMSE:", rmse(T_true, T_pred))
print("Td MAE:", mean_absolute_error(Td_true, Td_pred),
      "RMSE:", rmse(Td_true, Td_pred))
print("RH MAE:", mean_absolute_error(RH_true, RH_pred),
      "RMSE:", rmse(RH_true, RH_pred))
