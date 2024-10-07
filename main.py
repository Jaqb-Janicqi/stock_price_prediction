import streamlit as st
import pandas as pd
from datetime import datetime
from data_handling.yfiDownloader import download_sp500  # Ensure this path is correct
import os

# Function to download stock data from Stooq

def download_stooq_data(symbol):
    url = f"https://stooq.com/q/d/l/?s={symbol.lower()}&i=d"
    df = pd.read_csv(url)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

# Function to download and combine S&P 500 data for AAPL
def load_aapl_data():
    try:
        df_ticker = pd.read_csv(f'data/sp500/AAPL_1h.csv')  # Adjust this path if needed
        df_ticker['Ticker'] = 'AAPL'  # Add ticker column
        df_ticker['Date'] = pd.to_datetime(df_ticker['Date'])
        return df_ticker
    except FileNotFoundError:
        st.warning("No data found for AAPL. Please download it first.")
        return None

# Streamlit app
def main():
    st.title("Stock Data Viewer")

    # Sidebar: Select Stock Data
    selected_stock = st.sidebar.selectbox("Select Stock Data", ["AAPL", "SP500 (AAPL Only)"])

    if selected_stock == "AAPL":
        df = download_stooq_data("aapl.us")  
        min_date = df['Date'].min().date()
        max_date = df['Date'].max().date()

        # Sidebar: Date range slider for AAPL
        st.sidebar.header("Select Date Range for AAPL")
        start_date, end_date = st.sidebar.slider(
            "Date Range",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM-DD"
        )

        # Download AAPL data within the selected date range
        if st.button("Download AAPL Data"):
            st.write("Downloading AAPL data from Stooq...")
            try:
                start_date_ts = pd.Timestamp(start_date)
                end_date_ts = pd.Timestamp(end_date)
                df_filtered = df[(df['Date'] >= start_date_ts) & (df['Date'] <= end_date_ts)]
                st.write(f"Downloaded {len(df_filtered)} rows of AAPL stock data:")
                st.dataframe(df_filtered)

                if 'Close' in df_filtered.columns:
                    df_filtered.set_index('Date', inplace=True)
                    st.line_chart(df_filtered['Close'], width=700, height=300)

                # Option to download the data as CSV
                csv = df_filtered.to_csv(index=False).encode('utf-8')
                st.download_button(label="Download CSV", data=csv, file_name='AAPL_stooq_data.csv', mime='text/csv')

            except Exception as e:
                st.error(f"An error occurred: {e}")

    elif selected_stock == "SP500 (AAPL Only)":
        if st.button("Download SP500 Data for AAPL"):
            st.write("Downloading AAPL data from S&P 500...")
            try:
                # This will download the S&P 500 data for AAPL if not already downloaded
                download_sp500(interval='1h', split=True)

                # Load the combined data for AAPL
                aapl_data = load_aapl_data()

                # Ensure there is data available
                if aapl_data is not None:
                    # Print the AAPL data directly
                    st.write("AAPL Stock Data:")
                    st.dataframe(aapl_data)

                    sw.write(aapl_data)
                   

                st.success("AAPL data from S&P 500 loaded successfully!")

            except Exception as e:
                st.error(f"An error occurred while downloading AAPL data from SP500: {e}")

if __name__ == "__main__":
    main()
