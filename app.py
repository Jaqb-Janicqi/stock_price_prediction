import os
import pickle
import subprocess
import sys
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
from models.ARIMA_GARCH import ARIMA_GARCH
from models.RidgeRegression import RidgeRegression
from models.GRU import GRU
from models.LSTM_tower import LSTM_tower
from models.LitModel import LitModel
from setup import install_requirements, install_talib


install_talib()

# force cpu for torch
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


STATISTICAL_MODELS = ['ARIMA', 'ARIMA-GARCH', 'RidgeRegression']
ML_MODELS = ['LSTM', 'GRU']
ML_MODEL_DIR = 'best_ml_models'
STATISTICAL_MODEL_DIR = 'trained_statistical'
LIGHT_COLORS = [
    "#ADD8E6",  # Light Blue
    "#F08080",  # Light Coral
    "#E6E6FA",  # Lavender
    "#FFE4B5",  # Moccasin
    "#98FB98"   # Pale Green
]
DARK_COLORS = [
    "#00008B",  # Dark Blue
    "#8B0000",  # Dark Red
    "#4B0082",  # Indigo
    "#8B4513",  # Saddle Brown
    "#006400"   # Dark Green
]


def create_features(df: pd.DataFrame) -> None:
    from feature_creation import indicators
    indicators.add_candlestick_patterns(df)
    indicators.add_candlestick_patterns(df)
    indicators.add_moving_averages(df)


def create_candlestick_chart(df: pd.DataFrame, predictions: dict, ticker: str, plot=False) -> go.Figure:
    charts = []
    weekback = datetime.datetime.strptime(
        (datetime.datetime.now() - datetime.timedelta(days=7)).strftime('%Y-%m-%d'), '%Y-%m-%d').replace(tzinfo=datetime.timezone.utc)  # TODO simplify

    tmp_df = df.loc[weekback:]
    charts.append(go.Candlestick(
        x=tmp_df.index,
        open=tmp_df['Open'],
        high=tmp_df['High'],
        low=tmp_df['Low'],
        close=tmp_df['Close'],
        name='Stock Price'
    ))

    layout = go.Layout(
        title=f'{ticker} Stock Price',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
    )

    for idx, (key, value) in enumerate(predictions.items()):
        charts.append(go.Candlestick(
            x=value.index,
            open=value['Open'],
            high=value['High'],
            low=value['Low'],
            close=value['Close'],
            name=key,
            increasing_line_color=LIGHT_COLORS[idx % len(LIGHT_COLORS)],
            decreasing_line_color=DARK_COLORS[idx % len(DARK_COLORS)]
        ))

    fig = go.Figure(data=charts, layout=layout)
    if plot:
        fig.show(config={'modeBarButtonsToAdd': [
            'drawline',
            'drawopenpath',
            'drawclosedpath',
            'drawcircle',
            'drawrect',
            'eraseshape'
        ]})
    return fig


def get_dataset(df, input_size, output_size, transformation) -> pdat:
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


def train_arima(df) -> ARIMA:
    model = ARIMA(p=5, d=1, q=0)
    model.fit(df['Close'])
    return model


def train_arima_garch(df) -> ARIMA_GARCH:
    model = ARIMA_GARCH(p=5, d=1, q=0, p_garch=1, q_garch=1)
    model.fit(df['Close'])
    return model


def train_ridge_regression(df) -> RidgeRegression:
    df['Returns'] = (df['Close'].pct_change() * 100).round(2)
    df.dropna(inplace=True)
    model = RidgeRegression(alpha=1)
    model.history = list(df['Returns'])
    model.fit()
    return model


def predict_statistical_model(model_name, stock, df) -> float:
    if model_name == 'ARIMA':
        model = train_arima(df)
        return model.predict(1)
    elif model_name == 'ARIMA-GARCH':
        model = train_arima_garch(df)
        return model.predict(1)
    elif model_name == 'RidgeRegression':
        model = train_ridge_regression(df)
        return df['Close'].iloc[-1] * (1 + model.predict() / 100)


def load_gru() -> LitModel:
    gru = GRU(input_size=1, hidden_size=128, output_size=1, num_layers=4)
    lit_gru = LitModel.load_from_checkpoint(
        os.path.join(ML_MODEL_DIR, 'GRU_1_1_128_4_stationary'),
        model=gru
    )
    lit_gru.model.eval()
    lit_gru.transformation = 'stationary'
    return lit_gru


def load_lstm() -> LitModel:
    lstm_tower = LSTM_tower(input_size=61, output_size=4)
    lit_lstm_tower = LitModel.load_from_checkpoint(
        os.path.join(ML_MODEL_DIR, 'LSTM_tower_61_4_stationary'),
        model=lstm_tower
    )
    lit_lstm_tower.model.eval()
    lit_lstm_tower.transformation = 'stationary'
    return lit_lstm_tower


def load_ml_model(model_name) -> LitModel:
    if model_name == 'LSTM':
        return load_lstm()
    elif model_name == 'GRU':
        return load_gru()
    return None


def predict_ml_model(model_name, df) -> np.ndarray:
    model = load_ml_model(model_name)
    dataset = get_dataset(df, model.input_size,
                          model.output_size, model.transformation)
    pred = model(torch.tensor(dataset.last_X).unsqueeze(
        0).float()).cpu().detach().numpy()
    return dataset.last_true_candle + pred * dataset.last_true_candle


def wrangle_prediction(df, prediction: np.ndarray, index=None) -> pd.DataFrame:
    if prediction is None:
        return
    prediction = prediction.round(2)
    try:
        # hacky way to know if this is a statistical model
        # since shape access will throw an error
        if prediction.shape[1] > 1:
            new_row = np.concatenate(
                [np.array(prediction[0][:]), [df['Volume'].iloc[-1].item()]])
        else:
            new_row = np.array(prediction[0])
    except:
        new_row = np.array(prediction)
    if len(new_row.shape) == 1 and new_row.shape[0] == 5:
        columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    else:
        columns = ['Close']

    if index is not None:
        n_index = index
    else:
        n_index = [df.index[-1] + datetime.timedelta(hours=1)]
    predict_df = pd.DataFrame(
        [new_row], columns=columns, index=n_index)
    return predict_df


def get_prediction(model, stock, df, f_range) -> pd.DataFrame:
    if df is None:
        return None

    y_hat = df.tail(1).copy()
    for _ in range(f_range):
        if model in STATISTICAL_MODELS:
            y_hat = pd.concat(
                [y_hat, wrangle_prediction(df, predict_statistical_model(model, stock, pd.concat([df, y_hat])), index=[y_hat.index[-1] + datetime.timedelta(hours=1)])])

        elif model in ML_MODELS:
            y_hat = pd.concat(
                [y_hat, wrangle_prediction(df, predict_ml_model(model, pd.concat([df, y_hat])), index=[y_hat.index[-1] + datetime.timedelta(hours=1)])])
    return y_hat


def forecast_all(stock, f_range, start_date, end_date) -> tuple:
    df, predictions = None, None
    df = download_ticker(stock, '1h', cols=[
        'Open', 'High', 'Low', 'Close', 'Volume'])
    if df is None:
        st.write('No data found')
    else:
        # ensure there is enough data to calculate all indicators
        df = df.loc[start_date - datetime.timedelta(days=60):end_date]
        predictions = {}
        for model in STATISTICAL_MODELS + ML_MODELS:
            predictions[model] = get_prediction(model, stock, df, f_range)[1:]
            for col in predictions[model].columns:
                if predictions[model][col].isnull().values.any():
                    predictions[model][col] = predictions[model]['Close']

            predictions[model]['Low'] = predictions[model][['Open', 'High', 'Low', 'Close']].min(
                axis=1)
            predictions[model]['High'] = predictions[model][['Open', 'High', 'Low', 'Close']].max(
                axis=1)
    return df, predictions


def main():
    df, predictions, stock = None, None, None
    st.title('Stock Price Forecasting')
    st.sidebar.title('Stock Selection')

    pretrained_list = ['AAPL', 'GOOGL', 'MSFT', 'GME', 'AMZN', 'TSLA', 'FB', 'NFLX',
                       'NVDA', 'INTC', 'APO', 'BIO', 'FERG', 'CMA', 'DXC', 'ETSY', 'TTD', 'VEEV']
    pretrained_list.sort()
    st.write()
    selected_pretrained = st.sidebar.selectbox(
        'Select a stock from this list', [''] + pretrained_list)

    stock = st.sidebar.text_input(
        'or enter a stock ticker', value='AAPL').upper()

    f_range = st.sidebar.slider('Forecast range', 1, 30, 5, 1)

    if selected_pretrained != '':
        stock = selected_pretrained

    st.sidebar.write('')

    start_date = st.sidebar.text_input('Start Date', (datetime.datetime.now(
    ) - datetime.timedelta(days=60)).strftime('%Y-%m-%d'))
    end_date = st.sidebar.text_input(
        'End Date', datetime.datetime.now().strftime('%Y-%m-%d'))

    if 'small_chart' not in st.session_state:
        st.session_state.small_chart = False

    if st.sidebar.button('Forecast') or st.session_state.small_chart:
        print('here')
        start_date_dt = datetime.datetime.strptime(
            start_date, '%Y-%m-%d').replace(tzinfo=datetime.timezone.utc)
        end_date_dt = datetime.datetime.strptime(
            end_date, '%Y-%m-%d').replace(tzinfo=datetime.timezone.utc)
        df, predictions = forecast_all(
            stock, f_range, start_date_dt, end_date_dt)

        if 'small_chart' not in st.session_state:
            st.session_state.small_chart = False

        if not st.session_state.small_chart:
            st.plotly_chart(create_candlestick_chart(
                df, predictions, stock), use_container_width=True, key='small_chart')
            st.session_state.small_chart = True
        else:
            print('small chart')
            st.plotly_chart(create_candlestick_chart(
                df, predictions, stock, True), use_container_width=True, key='full_chart')

        if st.button('Full chart'):
            st.session_state.small_chart = not st.session_state.small_chart
    else:
        st.session_state.small_chart = False


def start_app_from_terminal():
    subprocess.check_call([sys.executable, "-m", "streamlit", "run", "app.py"])


if __name__ == '__main__':
    main()
