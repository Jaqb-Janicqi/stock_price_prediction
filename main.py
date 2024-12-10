import streamlit as st
import pandas as pd
from datetime import datetime
from data_handling.yfiDownloader import download_sp500  # Ensure this path is correct
import os
from time import sleep


import talib




# Streamlit app
def main():
    st.title("Stock Data Viewer")

    # Sidebar: Select Stock Data
    
    # Step 1: Select tickers
    st.header("Step 1: Select Tickers or Download All")
    sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    tickers = sp500['Symbol'].tolist()
    
    # Checkbox for selecting all tickers
    select_all = st.checkbox("Select all S&P 500 tickers")
    
    # Show the multi-select if "Select all" is not checked
    if select_all:
        selected_tickers = tickers  # If "Select all" is checked, select all tickers
        st.write("All S&P 500 tickers are selected.")
    else:
        selected_tickers = st.multiselect("Select one or more tickers", tickers)

    if st.button("Download Data"):
        st.write("downloading")
        download_sp500(selected_tickers,interval='1h', split=True)
        st.write("downloaded")
    show_data = st.checkbox("Do you want to see data")

    if show_data:
        filenames = [f for f in os.listdir('data/sp500') if os.path.isfile(os.path.join('data/sp500', f))]

        to_preview=st.selectbox("Preview dataset",filenames)
        chart = st.line_chart()
        data = pd.read_csv('data/sp500/'+to_preview, parse_dates=['Datetime'])
        
        # Set 'Datetime' as index
        data.set_index('Datetime', inplace=True)
        data["SMA"] = talib.SMA(data.Close, timeperiod=3)
        data["MA"]  = talib.MA(data.Close, timeperiod=3)
        data["EMA"] = talib.EMA(data.Close, timeperiod=3)
        data["WMA"] = talib.WMA(data.Close, timeperiod=3)
        data["RSI"] = talib.RSI(data.Close, timeperiod=3)
        data["MOM"] = talib.MOM(data.Close, timeperiod=3)
        data["DEMA"] = talib.DEMA(data.Close, timeperiod=3)
        data["TEMA"] = talib.TEMA(data.Close, timeperiod=3)
        chart.add_rows(data['Close'])
        st.write(data)
        
    

if __name__ == "__main__":
    main()