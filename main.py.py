# Import required modules
import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet  
from prophet.plot import plot_plotly
import plotly.graph_objs as go
import matplotlib.pyplot as plt

# Set Streamlit page config
st.set_page_config(page_title="TrendWise", layout="wide")

# Function to download stock data
def download_stock_data(stock_symbol, start, end):
    return yf.download(stock_symbol, start=start, end=end)

# Function to plot raw data
def plot_raw_data(stock_data, stock_symbol):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Open'], name=f'{stock_symbol} Open', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], name=f'{stock_symbol} Close', line=dict(color='red')))
    fig.update_layout(title="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

# Function to calculate moving averages
def calculate_moving_averages(stock_data, ma50=50, ma200=200):
    stock_data['MA50'] = stock_data['Close'].rolling(window=ma50).mean()
    stock_data['MA200'] = stock_data['Close'].rolling(window=ma200).mean()
    return stock_data

# Function to calculate RSI
def calculate_rsi(stock_data, window=14):
    delta = stock_data['Close'].diff()
    gain = delta.clip(lower=0).rolling(window=window).mean()
    loss = -delta.clip(upper=0).rolling(window=window).mean()
    rs = gain / loss
    stock_data['RSI'] = 100 - (100 / (1 + rs))
    return stock_data

# Function to forecast stock data using Prophet
def forecast_stock(stock_data, stock_symbol, period):
    df_train = stock_data.reset_index().rename(columns={"Date": "ds", "Close": "y"})[['ds', 'y']]
    model = Prophet()
    model.fit(df_train)
    future = model.make_future_dataframe(periods=period)
    forecast = model.predict(future)

    st.subheader(f"Forecast for {stock_symbol}")
    st.write(forecast.tail())

    fig1 = plot_plotly(model, forecast)
    st.plotly_chart(fig1)

    st.write(f"Forecast Components for {stock_symbol}")
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)

# Main Streamlit app
st.title("📈 TrendWise - Stock Trend Prediction Web App")

# Input: Stock tickers
stock_symbol1 = st.text_input("Enter Stock Ticker 1", "AAPL")
stock_symbol2 = st.text_input("Enter Stock Ticker 2", "MSFT")

# Date range
start_date = "2018-01-01"
end_date = date.today().strftime("%Y-%m-%d")

# Fetch stock data
stock_data1 = download_stock_data(stock_symbol1, start_date, end_date)
stock_data2 = download_stock_data(stock_symbol2, start_date, end_date)

# Display raw stock data
st.subheader("🔍 Raw Stock Data")
st.write(f"**{stock_symbol1}**", stock_data1.tail())
st.write(f"**{stock_symbol2}**", stock_data2.tail())

# Plot raw data
plot_raw_data(stock_data1, stock_symbol1)
plot_raw_data(stock_data2, stock_symbol2)

# Add technical indicators
stock_data1 = calculate_moving_averages(stock_data1)
stock_data2 = calculate_moving_averages(stock_data2)
stock_data1 = calculate_rsi(stock_data1)
stock_data2 = calculate_rsi(stock_data2)

# Plot moving averages
st.subheader("📊 Moving Averages")
fig_ma = go.Figure()
fig_ma.add_trace(go.Scatter(x=stock_data1.index, y=stock_data1['MA50'], name=f'{stock_symbol1} MA50', line=dict(color='green')))
fig_ma.add_trace(go.Scatter(x=stock_data1.index, y=stock_data1['MA200'], name=f'{stock_symbol1} MA200', line=dict(color='orange')))
fig_ma.add_trace(go.Scatter(x=stock_data2.index, y=stock_data2['MA50'], name=f'{stock_symbol2} MA50', line=dict(color='purple')))
fig_ma.add_trace(go.Scatter(x=stock_data2.index, y=stock_data2['MA200'], name=f'{stock_symbol2} MA200', line=dict(color='yellow')))
fig_ma.update_layout(title="Moving Averages", xaxis_rangeslider_visible=True)
st.plotly_chart(fig_ma)

# Plot RSI
st.subheader("📉 Relative Strength Index (RSI)")
fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(x=stock_data1.index, y=stock_data1['RSI'], name=f'{stock_symbol1} RSI', line=dict(color='blue')))
fig_rsi.add_trace(go.Scatter(x=stock_data2.index, y=stock_data2['RSI'], name=f'{stock_symbol2} RSI', line=dict(color='red')))
fig_rsi.update_layout(title="RSI", xaxis_rangeslider_visible=True)
st.plotly_chart(fig_rsi)

# Forecasting
n_years = st.slider("📅 Select years to forecast", 1, 4)
period = n_years * 365

forecast_stock(stock_data1, stock_symbol1, period)
forecast_stock(stock_data2, stock_symbol2, period)
