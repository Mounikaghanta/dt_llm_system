class AnomalyGate:
    """
    Gate decides whether DT assimilates sensor measurements (clean)
    or uses its own predictions (anomalous).
    """

    def update_state(self, dt, sensor_row, prediction, anomaly_label: int):
        # anomaly_label: 0 = clean, 1 = anomaly
        if anomaly_label == 0:
            dt.update_state(
                sensor_row["T (degC)"],
                sensor_row["Tdew (degC)"],
                sensor_row["rh (%)"],
            )
        else:
            dt.update_state(
                prediction["T_pred"],
                prediction["Td_pred"],
                prediction["RH_pred"],
            )
