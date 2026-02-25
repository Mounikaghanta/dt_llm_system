import numpy as np
import pandas as pd


class DigitalTwin:
    """
    Physics-guided local DT for a single weather station.
    Uses calibrated hourly baselines + smooth relaxation + physical constraints.
    """

    def __init__(self, baseline_path="hourly_baseline.csv", alpha=0.25, beta=0.05):
        self.trusted_state = None
        self.alpha = alpha  # how strongly we relax temperature toward diurnal baseline
        self.beta = beta    # how strongly dewpoint follows temperature change

        # hour -> expected temperature (from clean month)
        hourly = pd.read_csv(baseline_path, index_col=0)
        # handle either "T (degC)" column or unnamed series
        if "T (degC)" in hourly.columns:
            s = hourly["T (degC)"]
        else:
            s = hourly.iloc[:, 0]
        self.hourly_T = {int(k): float(v) for k, v in s.to_dict().items()}

    @staticmethod
    def vapor_pressure(T):
        return 6.11 * np.exp((17.27 * T) / (237.7 + T))

    def compute_rh(self, T, Td):
        e_T = self.vapor_pressure(T)
        e_Td = self.vapor_pressure(Td)
        RH = 100.0 * (e_Td / e_T)
        return float(np.clip(RH, 0, 100))

    def predict(self, timestamp):
        if self.trusted_state is None:
            raise ValueError("Trusted state not initialized.")

        T_prev = float(self.trusted_state["T"])
        Td_prev = float(self.trusted_state["Td"])

        hour = int(timestamp.hour)
        T_base = self.hourly_T.get(hour, T_prev)

        # Temperature: persistence + relaxation to calibrated diurnal baseline
        T_pred = T_prev + self.alpha * (T_base - T_prev)

        # Dew point: slow evolution following temperature change
        Td_pred = Td_prev + self.beta * (T_pred - T_prev)

        # Physics constraint
        Td_pred = min(Td_pred, T_pred)

        # RH derived from T and Td
        RH_pred = self.compute_rh(T_pred, Td_pred)

        return {"T_pred": T_pred, "Td_pred": Td_pred, "RH_pred": RH_pred}

    def update_state(self, T, Td, RH):
        # RH is not used as an independent state driver (it’s derived), but we store it anyway
        self.trusted_state = {"T": float(T), "Td": float(Td), "RH": float(RH)}
