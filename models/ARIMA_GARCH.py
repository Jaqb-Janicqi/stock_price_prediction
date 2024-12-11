import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA as StatsmodelsARIMA
from arch import arch_model as garch


class ARIMA_GARCH:
    def __init__(self, p, d, q, p_garch=1, q_garch=1):
        self.p = p
        self.d = d
        self.q = q
        self.p_garch = p_garch
        self.q_garch = q_garch
        self.model = None
        self.garch_model = None
        self.history = []

    def fit(self, data):
        # Fit the ARIMA model
        self.history = list(data)
        self.model = StatsmodelsARIMA(
            self.history, order=(self.p, self.d, self.q)).fit()
        self.garch_model = garch(
            self.model.resid, vol='GARCH', p=self.p_garch, q=self.q_garch).fit(disp='off')

    def predict(self, steps):
        # Make predictions
        if self.model is None or self.garch_model is None:
            raise ValueError("Model must be fitted first before prediction.")

        arima_forecast = self.model.forecast(steps)
        garch_forecast = self.garch_model.forecast(
            horizon=steps).mean.values[-1, 0]
        return arima_forecast + garch_forecast

    def rolling_forecast(self, test_data):
        predictions = []
        for actual in test_data:
            # Fit the model on the current history
            self.fit(pd.Series(self.history))
            forecast = self.predict(1)
            predictions.append(forecast)
            # Update history with actual value
            self.history.append(actual)
        return np.array(predictions)
