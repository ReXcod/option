import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime, timedelta

# --- Page Config ---
st.set_page_config(page_title="Black-Scholes Intelligence Engine", layout="wide")

# --- Custom CSS for "Hacker/Quant" Vibe ---
st.markdown("""
<style>
    .metric-container {
        background-color: #0E1117;
        border: 1px solid #262730;
        padding: 20px;
        border-radius: 10px;
    }
    .stMetricLabel {font-weight: bold; color: #FAFAFA;}
</style>
""", unsafe_allow_html=True)

# --- 1. The Math Class (Backend) ---
class BlackScholes:
    def __init__(self, S, K, T, r, sigma):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma

    def _d1_d2(self):
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        return d1, d2

    def call_price(self):
        d1, d2 = self._d1_d2()
        return (self.S * norm.cdf(d1)) - (self.K * np.exp(-self.r * self.T) * norm.cdf(d2))

    def put_price(self):
        d1, d2 = self._d1_d2()
        return (self.K * np.exp(-self.r * self.T) * norm.cdf(-d2)) - (self.S * norm.cdf(-d1))
    
    def calculate_greeks(self):
        d1, d2 = self._d1_d2()
        pdf_d1 = norm.pdf(d1)
        delta_call = norm.cdf(d1)
        gamma = pdf_d1 / (self.S * self.sigma * np.sqrt(self.T))
        vega = self.S * np.sqrt(self.T) * pdf_d1 / 100  # Scaled for %
        theta_call = (-(self.S * self.sigma * pdf_d1) / (2 * np.sqrt(self.T)) - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2)) / 365
        return delta_call, gamma, vega, theta_call

# --- 2. Sidebar Inputs ---
st.sidebar.header("1. Market Data")
ticker_input = st.sidebar.text_input("Ticker Symbol", value="NVDA")
try:
    # Fetch Live Data
    stock = yf.Ticker(ticker_input)
    history = stock.history(period="1y")
    current_price = history['Close'].iloc[-1]
    
    # Calculate Historical Volatility (Annualized)
    history['Log Returns'] = np.log(history['Close'] / history['Close'].shift(1))
    hist_vol = history['Log Returns'].std() * np.sqrt(252)
    
    st.sidebar.success(f"Live Price: ${current_price:.2f}")
    st.sidebar.info(f"Hist. Volatility: {hist_vol:.2%}")
except:
    st.sidebar.error("Invalid Ticker")
    current_price, hist_vol = 100, 0.2

st.sidebar.header("2. Option Specs")
# Smart Strike Selection
strike_input = st.sidebar.number_input("Strike Price ($)", value=float(round(current_price, 0)))
days_input = st.sidebar.number_input("Days to Expiration", value=30)
volatility_input = st.sidebar.slider("Volatility (Sigma)", 0.10, 1.50, float(hist_vol), 0.01)
risk_free_input = st.sidebar.number_input("Risk-Free Rate (Decimal)", value=0.045)

# --- 3. Main Dashboard Area ---
st.title(f"{ticker_input.upper()} Options Intelligence")
st.markdown("---")

# Initialize Engine
T = days_input / 365.0
bs = BlackScholes(current_price, strike_input, T, risk_free_input, volatility_input)
call_price = bs.call_price()
put_price = bs.put_price()
delta, gamma, vega, theta = bs.calculate_greeks()

# Row 1: The Big Numbers
col1, col2, col3, col4 = st.columns(4)
col1.metric("Theoretical Call Price", f"${call_price:.2f}")
col2.metric("Theoretical Put Price", f"${put_price:.2f}")
col3.metric("Model Volatility", f"{volatility_input:.1%}")
col4.metric("Days to Expiry", f"{days_input} days")

# Row 2: The Prediction Signal (Manual Input for Comparison)
st.markdown("### 🤖 Market Scanner (Arbitrage Detector)")
col_a, col_b = st.columns([1, 3])

with col_a:
    market_price = st.number_input("Enter Actual Market Price ($)", value=0.0, step=0.1)

with col_b:
    if market_price > 0:
        diff = market_price - call_price
        pct_diff = (diff / call_price) * 100
        
        if market_price < call_price:
            st.success(f"**SIGNAL: UNDERVALUED** (Discount: {abs(pct_diff):.1f}%) — Strong Buy Candidate 🟢")
            st.write(f"The market is charging ${market_price}, but math says it's worth ${call_price:.2f}.")
        else:
            st.error(f"**SIGNAL: OVERVALUED** (Premium: {pct_diff:.1f}%) — Avoid / Sell 🔴")
            st.write(f"The market is charging ${market_price}, but math says it's worth ${call_price:.2f}.")
    else:
        st.info("Enter the real-time option price from your broker to see the prediction.")

st.markdown("---")

# Row 3: Deep Analysis (Heatmap & Greeks)
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("🗺️ Scenario Analysis (Heatmap)")
    st.write("How does Call Price change if Stock Price (X-axis) or Volatility (Y-axis) moves?")
    
    # Generate Heatmap Data
    spot_range = np.linspace(current_price * 0.8, current_price * 1.2, 10)
    vol_range = np.linspace(volatility_input * 0.5, volatility_input * 1.5, 10)
    
    prices = np.zeros((10, 10))
    for i, v in enumerate(vol_range):
        for j, s in enumerate(spot_range):
            bs_temp = BlackScholes(s, strike_input, T, risk_free_input, v)
            prices[i, j] = bs_temp.call_price()
            
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(prices, xticklabels=np.round(spot_range, 1), yticklabels=np.round(vol_range, 2), annot=True, fmt=".1f", cmap="viridis", ax=ax)
    ax.set_xlabel("Stock Price ($)")
    ax.set_ylabel("Volatility")
    st.pyplot(fig)

with col_right:
    st.subheader("The Greeks")
    st.write("Risk sensitivity metrics.")
    
    greek_data = {
        "Metric": ["Delta (Δ)", "Gamma (Γ)", "Vega (ν)", "Theta (Θ)"],
        "Value": [f"{delta:.3f}", f"{gamma:.4f}", f"{vega:.3f}", f"{theta:.3f}"],
        "Meaning": [
            "Change per $1 stock move",
            "Acceleration of Delta",
            "Change per 1% Volatility",
            "Time decay per day"
        ]
    }
    st.table(pd.DataFrame(greek_data))
