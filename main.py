
#Importing the Required modules
import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet  
from prophet.plot import plot_plotly
from plotly import graph_objs as go



# Function to download stock data
def download_stock_data(stock_symbol, start, end):
    df = yf.download(stock_symbol, start=start, end=end)
    return df



# Main Streamlit app
st.title("TrendWise - A Stock Trend Prediction Web App")

# User input for stock symbols (two inputs)
stock_symbol1 = st.text_input('Enter Stock Ticker 1','AAPL')
stock_symbol2 = st.text_input('Enter Stock Ticker 2', 'MSFT')

# Download stock data for both symbols
stock_data1 = download_stock_data(stock_symbol1, "2018-01-01", date.today().strftime("%Y-%m-%d"))
stock_data2 = download_stock_data(stock_symbol2, "2018-01-01", date.today().strftime("%Y-%m-%d"))



# Display downloaded data for both stocks
st.subheader("Stock Data")
st.write("Stock Data for", stock_symbol1)
st.write(stock_data1.tail())
st.write("Stock Data for", stock_symbol2)
st.write(stock_data2.tail())


n_years=st.slider("Years of prediction",1,4)
period= n_years * 365



#Defining a function to plot raw data
def plot_raw_data(stock_data, stock_symbol):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Open'], name=stock_symbol + ' Open', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], name=stock_symbol + ' Close', line=dict(color='red')))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)



# Plot raw data for both stocks
st.subheader("Raw Stock Data")
plot_raw_data(stock_data1, stock_symbol1)
plot_raw_data(stock_data2, stock_symbol2)

# Calculate moving averages
def calculate_moving_averages(stock_data, window=50):
    stock_data['MA50'] = stock_data['Close'].rolling(window=window).mean()
    stock_data['MA200'] = stock_data['Close'].rolling(window=200).mean()
    return stock_data


# Add moving averages to stock data
stock_data1 = calculate_moving_averages(stock_data1)
stock_data2 = calculate_moving_averages(stock_data2)


# Plot moving averages
st.subheader("Moving Averages")
fig_ma = go.Figure()
fig_ma.add_trace(go.Scatter(x=stock_data1.index, y=stock_data1['MA50'], name=stock_symbol1 + ' MA50', line=dict(color='green')))
fig_ma.add_trace(go.Scatter(x=stock_data1.index, y=stock_data1['MA200'], name=stock_symbol1 + ' MA200', line=dict(color='orange')))
fig_ma.add_trace(go.Scatter(x=stock_data2.index, y=stock_data2['MA50'], name=stock_symbol2 + ' MA50', line=dict(color='purple')))
fig_ma.add_trace(go.Scatter(x=stock_data2.index, y=stock_data2['MA200'], name=stock_symbol2 + ' MA200', line=dict(color='yellow')))
fig_ma.layout.update(title_text="Moving Averages", xaxis_rangeslider_visible=True)
st.plotly_chart(fig_ma)


# Calculate Relative Strength Index (RSI)
def calculate_rsi(stock_data, window=14):
    delta = stock_data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    stock_data['RSI'] = rsi
    return stock_data


# Add RSI to stock data
stock_data1 = calculate_rsi(stock_data1)
stock_data2 = calculate_rsi(stock_data2)


# Plot RSI
st.subheader("Relative Strength Index (RSI)")
fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(x=stock_data1.index, y=stock_data1['RSI'], name=stock_symbol1 + ' RSI', line=dict(color='blue')))
fig_rsi.add_trace(go.Scatter(x=stock_data2.index, y=stock_data2['RSI'], name=stock_symbol2 + ' RSI', line=dict(color='red')))
fig_rsi.layout.update(title_text="RSI", xaxis_rangeslider_visible=True)
st.plotly_chart(fig_rsi)


#Forecasting data for stock 1
df_train1 = stock_data1.reset_index()[['Date', 'Close']]
df_train1 = df_train1.rename(columns={"Date": "ds", "Close": "y"})

m1 = Prophet()
m1.fit(df_train1)
future1 = m1.make_future_dataframe(periods=period)
forecast1 = m1.predict(future1)

st.subheader('Forecast data for ' + stock_symbol1)
st.write(forecast1.tail())

st.write('Forecast Data for ' + stock_symbol1)
fig1 = plot_plotly(m1, forecast1)
st.plotly_chart(fig1)

st.write('Forecast components for ' + stock_symbol1)
fig2 = m1.plot_components(forecast1)
st.write(fig2)


#Forecasting data for stock 2
df_train2 = stock_data2.reset_index()[['Date', 'Close']]
df_train2 = df_train2.rename(columns={"Date": "ds", "Close": "y"})

m2 = Prophet()
m2.fit(df_train2)
future2 = m2.make_future_dataframe(periods=period)
forecast2 = m2.predict(future2)

st.subheader('Forecast data for ' + stock_symbol2)
st.write(forecast2.tail())

st.write('Forecast Data for ' + stock_symbol2)
fig3 = plot_plotly(m2, forecast2)
st.plotly_chart(fig3)


st.write('Forecast components for ' + stock_symbol2)
fig4 = m2.plot_components(forecast2)
st.write(fig4)
