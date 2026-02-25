import math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

import joblib
import numpy as np


def _hour_sin_cos(ts: datetime):
    h = ts.hour + ts.minute / 60.0
    ang = 2.0 * math.pi * (h / 24.0)
    return math.sin(ang), math.cos(ang)


def _doy_sin_cos(ts: datetime):
    doy = ts.timetuple().tm_yday
    ang = 2.0 * math.pi * (doy / 365.0)
    return math.sin(ang), math.cos(ang)


def _sat_vapor_pressure_hpa(T_c: float) -> float:
    # Magnus (common in met): hPa
    return 6.112 * math.exp((17.62 * T_c) / (243.12 + T_c))


def rh_from_T_Td(T_c: float, Td_c: float) -> float:
    es = _sat_vapor_pressure_hpa(T_c)
    e = _sat_vapor_pressure_hpa(Td_c)
    rh = 100.0 * (e / max(es, 1e-6))
    return float(max(0.0, min(100.0, rh)))


@dataclass
class DTState:
    T: float
    Td: float
    RH: float


class DigitalTwinV2:
    """
    Physics-informed DT:
    - Learns coefficients for T and Td using cyclical features (hour, day-of-year) + autoregressive terms.
    - Enforces Td <= T.
    - Computes RH from T and Td via vapor pressure relation.
    """

    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)  # dict with keys "T_model", "Td_model"
        self.state: Optional[DTState] = None

    def update_state(self, T: float, Td: float, RH: float):
        self.state = DTState(float(T), float(Td), float(RH))

    def _features(self, ts: datetime) -> np.ndarray:
        assert self.state is not None, "DT state not initialized"
        hour_sin, hour_cos = _hour_sin_cos(ts)
        doy_sin, doy_cos = _doy_sin_cos(ts)

        # interactions (simple but effective)
        hum_x_hour_sin = self.state.RH * hour_sin
        hum_x_hour_cos = self.state.RH * hour_cos

        return np.array([
            self.state.T,
            self.state.Td,
            self.state.RH,
            hour_sin, hour_cos,
            doy_sin, doy_cos,
            hum_x_hour_sin, hum_x_hour_cos
        ], dtype=float)

    def predict(self, ts: datetime) -> Dict[str, float]:
        x = self._features(ts).reshape(1, -1)
        T_pred = float(self.model["T_model"].predict(x)[0])
        Td_pred = float(self.model["Td_model"].predict(x)[0])

        # physical constraint
        if Td_pred > T_pred:
            Td_pred = T_pred

        RH_pred = rh_from_T_Td(T_pred, Td_pred)

        return {"T_pred": T_pred, "Td_pred": Td_pred, "RH_pred": RH_pred}
