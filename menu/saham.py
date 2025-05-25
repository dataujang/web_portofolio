import streamlit as st
import yfinance as yf
import pandas as pd

symbol ="TLKM"

st.write("# Simple Stock Price App")
st.write(f"Show are the stock closing price and volume of {symbol}")


dt=yf.Ticker(symbol +".JK")
df=dt.history(period='1d', start='2021-01-01', end='2025-05-19')


# You can use a column just like st.sidebar:

#st.subheader("Volume Penumpang Harian")
#st.line_chart(df, x='Date', y='Close', use_container_width=True)

st.line_chart(df.Close, use_container_width=True)


# Or even better, call Streamlit functions inside a "with" block:
st.bar_chart(df.Volume,use_container_width=True)