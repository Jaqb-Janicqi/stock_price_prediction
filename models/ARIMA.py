import numpy as np
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA as StatsmodelsARIMA

class ARIMA:
    def __init__(self, p, d, q):
        self.p = p
        self.d = d
        self.q = q
        self.model = None
        self.history = []
        
    def fit(self, data):
        # Fit the ARIMA model
        self.history = list(data)
        self.model = StatsmodelsARIMA(self.history, order=(self.p, self.d, self.q)).fit()
        
    def predict(self, steps):
        # Make predictions
        if self.model is None:
            raise ValueError("Model must be fitted first before prediction.")
        
        forecast = self.model.forecast(steps)
        return forecast
    
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
            