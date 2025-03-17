import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet

# 🎯 Nifty 50 Stock Symbols
nifty50_stocks = {
    "Reliance Industries": "RELIANCE.NS",
    "Tata Consultancy Services": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "State Bank of India": "SBIN.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "ITC": "ITC.NS",
    "Larsen & Toubro": "LT.NS",
    "Axis Bank": "AXISBANK.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "HCL Technologies": "HCLTECH.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "UltraTech Cement": "ULTRACEMCO.NS",
    "Titan Company": "TITAN.NS",
    "Sun Pharmaceutical": "SUNPHARMA.NS",
    "Wipro": "WIPRO.NS",
    "Mahindra & Mahindra": "M&M.NS",
}

# Streamlit UI
st.title("📈 Nifty 50 Stock Price Prediction using AI")
st.write("Select a Nifty 50 company and get future price predictions!")

# **Dropdown for Stock Selection**
stock_name = st.selectbox("Select a Stock:", list(nifty50_stocks.keys()))
stock_symbol = nifty50_stocks[stock_name]

# **Dropdown for Prediction Duration**
days_to_predict = st.selectbox("Select Prediction Duration:", [1, 7, 30, 365])

# Fetch stock data
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(period="5y")
    df.reset_index(inplace=True)
    df['Date'] = df['Date'].dt.tz_localize(None)  # ✅ Remove timezone issue
    return df

# Train Prophet model
def train_prophet(df):
    df = df[['Date', 'Close']]
    df.columns = ['ds', 'y']
    model = Prophet()
    model.fit(df)
    return model

# Make future predictions
def predict_future(model, days):
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    return forecast

# Run prediction when user clicks the button
if st.button("🔮 Predict Future Prices"):
    try:
        # Fetch Data
        data = get_stock_data(stock_symbol)
        
        # Train Model
        model = train_prophet(data)
        
        # Predict
        forecast = predict_future(model, days_to_predict)
        
        # Extract Prices
        last_price = data['Close'].iloc[-1]  # ✅ Last Closing Price
        predicted_price = forecast['yhat'].iloc[-1]  # ✅ Predicted Price
        min_price = forecast['yhat_lower'].iloc[-1]  # ✅ Minimum Price
        max_price = forecast['yhat_upper'].iloc[-1]  # ✅ Maximum Price

        # ✅ Display Predicted Results
        st.success(f"📌 **Last Closing Price:** ₹{last_price:.2f}")
        st.info(f"📈 **Predicted Price (in {days_to_predict} days):** ₹{predicted_price:.2f}")
        st.write(f"🟢 **Minimum Expected Price:** ₹{min_price:.2f}")
        st.write(f"🔴 **Maximum Expected Price:** ₹{max_price:.2f}")

        # ✅ Plot Interactive Graph with Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Historical Prices', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Predicted Prices', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Min Price', line=dict(color='green', dash='dot')))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Max Price', line=dict(color='purple', dash='dot')))
        fig.update_layout(title=f"Stock Price Prediction for {stock_name}", xaxis_title="Date", yaxis_title="Stock Price (₹)", legend_title="Legend")
        st.plotly_chart(fig)  # ✅ Show Plotly Chart

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
