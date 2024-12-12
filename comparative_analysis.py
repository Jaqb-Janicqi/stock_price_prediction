import os
import pandas as pd
import matplotlib.pyplot as plt

def load_last_30_points(filepath):
    data = pd.read_csv(filepath)
    return data.tail(60)

DATA_PATH = os.getcwd()
results_directory = 'results\\'
results_directory2 = 'results_st\\'

arima_garch_file = os.path.join(results_directory2, DATA_PATH + '\\results_st\\APO_arima_garch_results.csv')
lstm_file = os.path.join(results_directory, DATA_PATH + '\\results\\results_lstm_tower_APO_1h.csv')
gru_file = os.path.join(results_directory, DATA_PATH + '\\results\\results_gru_APO_1h.csv')

arima_garch_data = load_last_30_points(arima_garch_file)
lstm_data = load_last_30_points(lstm_file)
gru_data = load_last_30_points(gru_file)


true_price = lstm_data['y'].values 
lstm_predicted = lstm_data['pred'].values

gru_predicted = gru_data['pred'].values
arima_garch_predicted = arima_garch_data['Predicted'].values

mse_arima = ((true_price - arima_garch_predicted) ** 2).mean()
mse_lstm = ((true_price - lstm_predicted) ** 2).mean()
print('MSE ARIMA-GARCH:', mse_arima)
print('MSE LSTM:', mse_lstm)
# calculate prediction in reverse order
# tmp = []
# idx = len(gru_predicted) - 1
# while len(tmp) < 30:
#     tmp.append(float(true_price[idx-1]) + float(gru_predicted[idx-1]) * float(true_price[idx-1]))
#     idx -= 1
# gru_predicted = tmp[::-1]

plt.figure(figsize=(12, 6))

plt.plot(range(len(true_price)), true_price, label='True price', color='blue')
plt.plot(range(len(arima_garch_predicted)), arima_garch_predicted, label='ARIMA-GARCH Predicted', color='purple', linestyle='--')
plt.plot(range(len(lstm_predicted)), lstm_predicted, label='LSTM Predicted', color='green', linestyle='--')
plt.plot(range(len(gru_predicted)), gru_predicted, label='GRU Predicted', color='red', linestyle='--')

plt.title('Visualization of selected models on DXC stock')
plt.xlabel('Time Steps (Last 30 Points)')
plt.ylabel('Stock Price')
plt.legend()
plt.grid(True)
plt.show()