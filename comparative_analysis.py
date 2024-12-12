import os
import pandas as pd
import matplotlib.pyplot as plt

def load_last_30_points(filepath):
    data = pd.read_csv(filepath)
    return data.tail(30)

DATA_PATH = 'C:/Users/hubci/Desktop/Politechnika Gdanska/inzynierka/Pracka/Thesis/stock_price_prediction/'
results_directory = 'results/'
results_directory2 = 'results_st/'

arima_garch_file = os.path.join(results_directory2, DATA_PATH + 'results_st/APO_arima_garch_results.csv')
lstm_file = os.path.join(results_directory, DATA_PATH + 'results/results_lstm_tower_APO_1h.csv')
gru_file = os.path.join(results_directory, DATA_PATH + 'results/results_gru_APO_1h.csv')

arima_garch_data = load_last_30_points(arima_garch_file)
lstm_data = load_last_30_points(lstm_file)
gru_data = load_last_30_points(gru_file)

arima_garch_actual = arima_garch_data['Actual'].values 
arima_garch_predicted = arima_garch_data['Predicted'].values

lstm_actual = lstm_data['y'].values 
lstm_predicted = lstm_data['pred'].values

gru_actual = gru_data['y'].values
gru_predicted = gru_data['pred'].values


plt.figure(figsize=(12, 6))

plt.plot(range(len(arima_garch_actual)), arima_garch_actual, label='ARIMA-GARCH Actual', color='blue', linestyle='--')
plt.plot(range(len(arima_garch_predicted)), arima_garch_predicted, label='ARIMA-GARCH Predicted', color='blue')

plt.plot(range(len(lstm_actual)), lstm_actual, label='LSTM Actual', color='green', linestyle='--')
plt.plot(range(len(lstm_predicted)), lstm_predicted, label='LSTM Predicted', color='green')

plt.plot(range(len(gru_actual)), gru_actual, label='GRU Actual', color='red', linestyle='--')
plt.plot(range(len(gru_predicted)), gru_predicted, label='GRU Predicted', color='red')

plt.title('Visualization of selected models on DXC stock')
plt.xlabel('Time Steps (Last 30 Points)')
plt.ylabel('Stock Price')
plt.legend()
plt.grid(True)
plt.show()