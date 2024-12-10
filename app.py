import os
import streamlit as st
import pandas as pd
from data_handling.yfiDownloader import download_ticker
import plotly.graph_objs as go
import time
import datetime
from data_handling.pandasDataSet import PandasDataSet
from models.GRU import GRU
from models.LSTM_tower import LSTM_tower
from models.LitModel import LitModel


STATISTICAL_MODELS = ['ARIMA', 'ARIMA-GARCH', 'RidgeRegression']
ML_MODELS = ['LSTM', 'GRU']
ML_MODEL_DIR = 'best_ml_models'
STATISTICAL_MODEL_DIR = 'trained_statistical'


def create_candlestick_chart(df, ticker, start_date, end_date):
    tmp_df = df[(df.index >= start_date) & (df.index <= end_date)]
    candle_chart = go.Candlestick(
        x=tmp_df.index,
        open=tmp_df['Open'],
        high=tmp_df['High'],
        low=tmp_df['Low'],
        close=tmp_df['Close']
    )
    del tmp_df
    layout = go.Layout(
        title=f'{ticker} Stock Price',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=True
    )

    fig = go.Figure(data=[candle_chart], layout=layout)
    fig.update_yaxes(autorange=True)
    return fig



def load_statistical_model(model_name, stock, df):
    pass


def load_ml_model(model_name):
    if model_name == 'LSTM':
        lstm_tower = LSTM_tower(input_size=61, output_size=4)
        lit_lstm_tower = LitModel.load_from_checkpoint(
            os.path.join(ML_MODEL_DIR, 'LSTM_tower_61_4_stationary'),
            model=lstm_tower
        )
        return lit_lstm_tower
    
    elif model_name == 'GRU':
        gru = GRU(input_size=1, hidden_size=128, output_size=1, num_layers=4)
        lit_gru = LitModel.load_from_checkpoint(
            os.path.join(ML_MODEL_DIR, 'GRU_1_1_128_4_stationary'),
            model=gru
        )
        return lit_gru
    
    else:
        return None

def predict_statistical_model(model_name, stock, df):
    pass


def predict_ml_model(model_name, stock, df):
    pass


def get_prediction(model, stock, df):
    if model in STATISTICAL_MODELS:
        return predict_statistical_model(model, stock, df)
    elif model in ML_MODELS:
        return predict_ml_model(model, stock, df)
    else:
        return None


def main():
    st.title('Stock Price App')
    st.sidebar.title('Stock Selection')

    pretrained_list = ['AAPL', 'GOOGL', 'MSFT', 'GME', 'AMZN', 'TSLA', 'FB', 'NFLX',
                       'NVDA', 'INTC', 'APO', 'BIO', 'FERG', 'CMA', 'DXC', 'ETSY', 'TTD', 'VEEV']
    pretrained_list.sort()
    st.write()
    selected_pretrained = st.sidebar.selectbox(
        'Statistical models need to be fitted, before they can be used. You can select one of these stocks to use a pretrained model.', [''] + pretrained_list)

    stock = st.sidebar.text_input(
        'or enter a stock ticker. Statistical models will be trained automatically if selected. Each may take up to 10 minutes.')

    if selected_pretrained != '':
        stock = selected_pretrained

    st.sidebar.write('')

    start_date = st.sidebar.text_input('Start Date', (datetime.datetime.now(
    ) - datetime.timedelta(days=60)).strftime('%Y-%m-%d'))
    end_date = st.sidebar.text_input(
        'End Date', value=datetime.datetime.now().strftime('%Y-%m-%d'))

    model = st.sidebar.selectbox(
        'Select a model', STATISTICAL_MODELS + ML_MODELS)

    if st.sidebar.button('Submit'):
        df = download_ticker(stock, '1h', cols=[
                             'Open', 'High', 'Low', 'Close'])
        if df is None:
            st.write('No data found')
        else:

            prediction = get_prediction(model, stock, df)

            st.plotly_chart(create_candlestick_chart(
                df, stock, start_date, end_date), use_container_width=True)


if __name__ == '__main__':
    main()
