import pickle
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA as StatsmodelsARIMA
import matplotlib.pyplot as plt

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
            raise ValueError("Model must be fitted firstly before the prediction.")
        
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
    
# list of tested stock tickers
tickers = ['DXC','TTD','APO','CMA','ETSY','VEEV','BIO','FERG']

# directory paths
data_directory = 'data_ohlcv/'
output = 'trained_st_new'
os.makedirs(output, exist_ok=True)

for ticker in tickers:
    print('Processing', ticker)
    try:
        # Load training and testing data
        path = os.path.join(data_directory, f'{ticker}_1h.csv')
        data = pd.read_csv(path , low_memory=False)
        data['Datetime'] = pd.to_datetime(data['Datetime'])
        data.set_index('Datetime', inplace=True)

        train_size = int(len(data)*0.7)
        train_price = data['Close'][:train_size]
        test_price = data['Close'][train_size:]

        # Initialize and fit the model
        arima_model = ARIMA(p=5, d=1, q=0)

        # Fit the model on training data
        arima_model.fit(train_price)

        # Make predictions on the test set
        predictions = arima_model.rolling_forecast(test_price)

        # Calculate Mean Squared Error
        mse = mean_squared_error(test_price, predictions)
        print(f'MSE for {ticker}: {mse}')

        # Plot the results
        plt.plot(data.index[:train_size], train_price, color="blue", label="Train")
        plt.plot(data.index[train_size:], test_price, color="grey", label="Test")
        plt.plot(data.index[train_size:], predictions, label="Predicted", color='red', ls=':')
        plt.title(f"ARIMA Model Prediction for {ticker}")
        plt.legend()
        plt.show()
        plt.close()

        filename = os.path.join(
            output,
            f'ARIMA_{ticker}_{arima_model.p}-{arima_model.d}-{arima_model.q}_mse-{mse:.5f}.ckpt'
        )

        print(f'Saving model to {filename}')

        with open(filename, 'wb') as file:
            pickle.dump(arima_model, file)
        print(f"Model successfully saved to {filename}")
    except Exception as e:
            print(f"Error saving the model: {e}")