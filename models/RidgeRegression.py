import numpy as np
from sklearn.linear_model import Ridge

class RidgeRegression:
    def __init__(self, alpha:float):
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)
        self.history = []
        
    def fit(self):
        x_train = np.arange(len(self.history)).reshape(-1,1)
        y_train = np.array(self.history)
        self.model.fit(x_train,y_train)
        
    def predict(self):
        if not self.history:
            raise ValueError("Model must be fitted first before prediction.")
        
        x_prediction = np.array([[len(self.history)]])
        y_prediction = self.model.predict(x_prediction) 
        return y_prediction[0]
        
    def rolling_forecast(self, test_data):
        predictions =[]
        for actual in test_data:
            self.fit()
            forecast = self.predict()
            predictions.append(forecast)
            self.history.append(actual)
        return np.array(predictions)