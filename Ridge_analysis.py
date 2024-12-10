import pickle
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

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
            raise ValueError("Model must be fitted firstly before the prediction.")
        
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
        data = pd.read_csv(path, low_memory=False)
        data['Datetime'] = pd.to_datetime(data['Datetime'])
        data.set_index('Datetime', inplace=True)

        # Calculate the Daily Percentage Change in stock price - returns
        data['Returns'] = (data['Close'].pct_change() * 100).round(2)
        data.dropna(inplace=True)
        data.head()

        train_size = int(len(data)*0.7)
        train_price = data['Close'][:train_size]
        test_price = data['Close'][train_size:]

        ridge_model = RidgeRegression(alpha=1)
        ridge_model.history = list(data['Returns'][0:train_size])

        predicted_returns = ridge_model.rolling_forecast(data['Returns'][train_size:])

        predicted_prices = []
        for i, predicted_return in enumerate(predicted_returns):
            new_price = test_price.iloc[i - 1] if i > 0 else train_price.iloc[-1]
            predicted_prices.append(new_price * (1 + predicted_return / 100))
            
        # Calculate Mean Squared Error
        mse = mean_squared_error(test_price, predicted_prices)
        print(f'MSE: {mse}')

        # Plot the results
        plt.plot(data.index[:train_size], train_price, color="blue", label="Train")
        plt.plot(data.index[train_size:], test_price, color="grey", label="Test")
        plt.plot(data.index[train_size:], predicted_prices, label="Predicted", color='red', ls=':')
        plt.title("Ridge Regression Model Prediction")
        plt.legend()
        plt.show()

        filename = os.path.join(
            output,
            f'RidgeRegression_{ticker}_{ridge_model.alpha}_mse-{mse:.5f}.ckpt'
        )

        print(f'Saving model to {filename}')

        with open(filename, 'wb') as file:
            pickle.dump(ridge_model, file)
        print(f"Model successfully saved to {filename}")
                
    except Exception as e:
        print(f"Error saving the model: {e}")