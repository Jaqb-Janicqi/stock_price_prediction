import os
import pickle
import numpy as np
import streamlit as st
import pandas as pd
import torch
from data_handling.yfiDownloader import download_ticker
import plotly.graph_objs as go
import time
import datetime
from data_handling.pandasDataSet import PandasDataset as pdat
from models.ARIMA import ARIMA
from models.RidgeRegression import RidgeRegression
from models.GRU import GRU
from models.LSTM_tower import LSTM_tower
from models.LitModel import LitModel

# force cpu for torch
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


STATISTICAL_MODELS = ['ARIMA', 'ARIMA-GARCH', 'RidgeRegression']
ML_MODELS = ['LSTM', 'GRU']
ML_MODEL_DIR = 'best_ml_models'
STATISTICAL_MODEL_DIR = 'trained_statistical'


def create_features(df: pd.DataFrame) -> None:
    from feature_creation import indicators
    indicators.add_candlestick_patterns(df)
    indicators.add_candlestick_patterns(df)
    indicators.add_moving_averages(df)


def create_candlestick_chart(df, prediction, ticker, start_date, end_date):
    tmp_df = df[(df.index >= start_date) & (df.index <= end_date)]
    candle_chart = go.Candlestick(
        x=tmp_df.index,
        open=tmp_df['Open'],
        high=tmp_df['High'],
        low=tmp_df['Low'],
        close=tmp_df['Close'],
        name='Stock Price'
    )
    del tmp_df
    layout = go.Layout(
        title=f'{ticker} Stock Price',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
    )
    higlight = go.Candlestick(
        x=prediction.index,
        open=prediction['Open'],
        high=prediction['High'],
        low=prediction['Low'],
        close=prediction['Close'],
        name='Prediction',
        increasing_line_color='#FFFFFF'.lower(),
        decreasing_line_color='#FFFFFF'.lower()
    )
    fig = go.Figure(data=[candle_chart, higlight], layout=layout)
    return fig


def get_dataset(df, input_size, output_size, transformation):
    stationary_transform = True if transformation == 'stationary' else False
    normalize = True if transformation == 'normalize' else False
    if input_size == 1:
        cols = ['Close']
    else:
        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if output_size == 1:
        target_cols = ['Close']
    else:
        target_cols = ['Open', 'High', 'Low', 'Close']

    dataset = pdat(df, 30, cols=cols, target_cols=target_cols,
                   stationary_tranform=stationary_transform, normalize=normalize)
    if input_size > 5:
        create_features(dataset.dataframe)
        dataset.columns = dataset.dataframe.columns.tolist()
    dataset.apply_transform()
    return dataset


def train_arima(df):
    model = ARIMA(p=5, d=1, q=0)
    model.fit(df['Close'])
    return model


def train_arima_garch(df):
    return None


def train_ridge_regression(df):
    df['Returns'] = (df['Close'].pct_change() * 100).round(2)
    df.dropna(inplace=True)
    model = RidgeRegression(alpha=1)
    model.history = list(df['Returns'])
    model.fit()
    return model


def predict_statistical_model(model_name, stock, df):
    if model_name == 'ARIMA':
        model = train_arima(df)
        return model.predict(1)
    elif model_name == 'ARIMA-GARCH':
        model = train_arima_garch(df)
        return model.predict(1)
    elif model_name == 'RidgeRegression':
        model = train_ridge_regression(df)
        return df['Close'].iloc[-1] * (1 + model.predict() / 100)


def load_gru():
    gru = GRU(input_size=1, hidden_size=128, output_size=1, num_layers=4)
    lit_gru = LitModel.load_from_checkpoint(
        os.path.join(ML_MODEL_DIR, 'GRU_1_1_128_4_stationary'),
        model=gru
    )
    lit_gru.model.eval()
    lit_gru.transformation = 'stationary'
    return lit_gru


def load_lstm():
    lstm_tower = LSTM_tower(input_size=61, output_size=4)
    lit_lstm_tower = LitModel.load_from_checkpoint(
        os.path.join(ML_MODEL_DIR, 'LSTM_tower_61_4_stationary'),
        model=lstm_tower
    )
    lit_lstm_tower.model.eval()
    lit_lstm_tower.transformation = 'stationary'
    return lit_lstm_tower


def load_ml_model(model_name):
    if model_name == 'LSTM':
        return load_lstm()
    elif model_name == 'GRU':
        return load_gru()
    return None


def predict_ml_model(model_name, df):
    model = load_ml_model(model_name)
    dataset = get_dataset(df, model.input_size,
                          model.output_size, model.transformation)
    pred = model(torch.tensor(dataset.last_X).unsqueeze(
        0).float()).cpu().detach().numpy()
    return dataset.last_true_candle + pred * dataset.last_true_candle


def get_prediction(model, stock, df):
    if model in STATISTICAL_MODELS:
        return predict_statistical_model(model, stock, df)
    elif model in ML_MODELS:
        return predict_ml_model(model, df)
    else:
        return None


def wrangle_prediction(df, prediction: np.ndarray):
    if prediction is None:
        return
    prediction = prediction.round(2)
    try:
        # hacky way to know if this is a statistical model that predicts close
        # since shape access will throw an error
        if prediction.shape[1] == 1:
            new_row = np.full(4, prediction[0][0])
        else:
            new_row = np.array(prediction[0][:])
    except:
        new_row = np.full(4, prediction)

    n_index = [df.index[-1] + datetime.timedelta(hours=1)]
    predict_df = pd.DataFrame(
        [new_row], columns=['Open', 'High', 'Low', 'Close'], index=n_index)
    return predict_df


def main():
    st.title('Stock Price Forecasting')
    st.sidebar.title('Stock Selection')

    pretrained_list = ['AAPL', 'GOOGL', 'MSFT', 'GME', 'AMZN', 'TSLA', 'FB', 'NFLX',
                       'NVDA', 'INTC', 'APO', 'BIO', 'FERG', 'CMA', 'DXC', 'ETSY', 'TTD', 'VEEV']
    pretrained_list.sort()
    st.write()
    selected_pretrained = st.sidebar.selectbox('Select a stock from this list', [''] + pretrained_list)

    stock = st.sidebar.text_input(
        'or enter a stock ticker', value='AAPL').upper()

    if selected_pretrained != '':
        stock = selected_pretrained

    st.sidebar.write('')

    start_date = st.sidebar.text_input('Start Date', (datetime.datetime.now(
    ) - datetime.timedelta(days=60)).strftime('%Y-%m-%d'))
    end_date = st.sidebar.text_input(
        'End Date', value=datetime.datetime.now().strftime('%Y-%m-%d'))

    model = st.sidebar.selectbox(
        'Select a model. LSTM predicts an entire future candle, others predict close price.', STATISTICAL_MODELS + ML_MODELS)

    if st.sidebar.button('Submit'):
        df = download_ticker(stock, '1h', cols=[
                             'Open', 'High', 'Low', 'Close', 'Volume'])
        if df is None:
            st.write('No data found')
        else:
            prediction = wrangle_prediction(
                df, get_prediction(model, stock, df.copy()))

            st.plotly_chart(create_candlestick_chart(
                df, prediction, stock, start_date, end_date), use_container_width=True)


if __name__ == '__main__':
    main()
