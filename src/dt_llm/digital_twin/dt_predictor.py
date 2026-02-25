import numpy as np


class DigitalTwin:

    def __init__(self):
        self.trusted_state = None

    # -----------------------------
    # Magnus formula for vapor pressure
    # -----------------------------
    @staticmethod
    def vapor_pressure(T):
        return 6.11 * np.exp((17.27 * T) / (237.7 + T))

    def compute_rh(self, T, Td):
        e_T = self.vapor_pressure(T)
        e_Td = self.vapor_pressure(Td)
        RH = 100 * (e_Td / e_T)
        return np.clip(RH, 0, 100)

    # -----------------------------
    # Diurnal baseline temperature
    # -----------------------------
    def diurnal_temperature(self, hour, T_prev):
        # Expected amplitude ~5°C typical daily variation
        A = 5.0

        # Phase shift so peak around 14:00
        phase_shift = -2

        baseline = A * np.sin(2 * np.pi * (hour + phase_shift) / 24)

        return T_prev * 0.95 + 0.05 * baseline

    # -----------------------------
    # One-step prediction
    # -----------------------------
    def predict(self, timestamp):

        if self.trusted_state is None:
            raise ValueError("Trusted state not initialized.")

        T_prev = self.trusted_state["T"]
        Td_prev = self.trusted_state["Td"]

        hour = timestamp.hour

        # --- Temperature prediction ---
        T_diurnal = self.diurnal_temperature(hour, T_prev)

        # Smooth relaxation toward diurnal baseline
        T_pred = T_prev + 0.2 * (T_diurnal - T_prev)

        # --- Dew point evolution (slow) ---
        Td_pred = Td_prev + 0.05 * (T_pred - T_prev)

        # Enforce physical constraint
        Td_pred = min(Td_pred, T_pred)

        # --- RH from thermodynamics ---
        RH_pred = self.compute_rh(T_pred, Td_pred)

        return {
            "T_pred": T_pred,
            "Td_pred": Td_pred,
            "RH_pred": RH_pred
        }

    # -----------------------------
    # Update trusted state
    # -----------------------------
    def update_state(self, T, Td, RH):
        self.trusted_state = {
            "T": T,
            "Td": Td,
            "RH": RH
        }
