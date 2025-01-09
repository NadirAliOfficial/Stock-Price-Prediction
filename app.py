# app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Stock Analysis & Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 Stock Price Analysis and Prediction")

# --- SIDEBAR SETTINGS ---
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL")
start_date = st.sidebar.date_input("Start Date", datetime.date(2015, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())
forecast_steps = st.sidebar.slider("Forecast Steps (Days)", min_value=1, max_value=60, value=30)

# --- FETCH DATA ---
@st.cache_data
def load_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    average_gain = gain.rolling(window=14).mean()
    average_loss = loss.rolling(window=14).mean()
    rs = average_gain / average_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    return data.dropna()

data = load_stock_data(ticker, start_date, end_date)

# --- DISPLAY DATA ---
st.write("### 📊 Historical Data")
st.dataframe(data.tail())

# --- PLOT CLOSING PRICE ---
st.write("### 📈 Closing Price with Moving Averages")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(data.index, data['Close'], label='Close Price')
ax.plot(data.index, data['SMA_50'], label='50-day SMA')
ax.plot(data.index, data['SMA_200'], label='200-day SMA')
ax.legend()
st.pyplot(fig)

# --- PLOT RSI ---
st.write("### 📉 Relative Strength Index (RSI)")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(data.index, data['RSI'], label='RSI', color='orange')
ax.axhline(70, color='red', linestyle='--')
ax.axhline(30, color='green', linestyle='--')
ax.legend()
st.pyplot(fig)

# --- FORECAST FUTURE PRICES ---
st.write("### 🚀 Forecast Future Prices")
model_file = "lstm_stock_model.h5"

@st.cache_resource
def load_trained_model():
    return load_model(model_file)

if st.sidebar.button("Predict Future Prices"):
    try:
        model = load_trained_model()
        
        scaler = MinMaxScaler()
        features = ['Close', 'SMA_50', 'SMA_200', 'RSI']
        scaled_data = scaler.fit_transform(data[features])

        time_steps = 60
        last_time_steps = scaled_data[-time_steps:]
        last_time_steps = np.expand_dims(last_time_steps, axis=0)

        def recursive_forecast(model, last_time_steps, scaler, forecast_steps, feature_count):
            predictions_inv = []
            current_input = last_time_steps.copy()
            
            for _ in range(forecast_steps):
                pred = model.predict(current_input)
                pred_value = pred[0, 0]
                last_known = current_input[0, -1, 1:]
                pred_full = np.concatenate(([pred_value], last_known))
                pred_inv = scaler.inverse_transform(pred_full.reshape(1, -1))[:, 0]
                predictions_inv.append(pred_inv[0])
                new_step = np.concatenate(([pred_value], last_known)).reshape(1, 1, feature_count)
                current_input = np.concatenate((current_input[:, 1:, :], new_step), axis=1)
            
            return predictions_inv

        forecasted_prices = recursive_forecast(model, last_time_steps, scaler, forecast_steps, scaled_data.shape[1])
        forecasted_prices = np.array(forecasted_prices)
        
        future_dates = pd.date_range(data.index[-1] + pd.Timedelta(days=1), periods=forecast_steps)
        future_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': forecasted_prices})
        future_df.set_index('Date', inplace=True)
        
        st.write("### 📅 Predicted Prices")
        st.dataframe(future_df)
        
        st.write("### 📊 Forecast Plot")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(data.index, data['Close'], label='Historical Close Price')
        ax.plot(future_df.index, future_df['Predicted Close'], label='Predicted Close Price', color='green')
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred: {e}")

# --- HIGHLIGHTED PERIOD PLOT ---
# Define the full download period
start_date = '2024-01-01'
end_date = '2025-01-01'

# Download AAPL stock data
data = yf.download(ticker, start=start_date, end=end_date)

if data.empty:
    st.write("No data found for the specified date range.")
else:
    # Sidebar input for highlight period
    st.write("### 📅 Highlighted Period Plot")
    highlight_start = st.sidebar.date_input("Highlight Start Date", datetime.date(2024, 12, 1))
    highlight_end = st.sidebar.date_input("Highlight End Date", datetime.date(2025, 1, 12))
    
    # Ensure date inputs are within the dataset's date range
    highlight_start = pd.Timestamp(highlight_start, tz=data.index.tz)
    highlight_end = pd.Timestamp(highlight_end, tz=data.index.tz)

    # Split the data into highlighted and non-highlighted parts
    within_highlight = data[(data.index >= highlight_start) & (data.index <= highlight_end)]
    outside_highlight_before = data[data.index < highlight_start]
    outside_highlight_after = data[data.index > highlight_end]

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot outside highlighted regions in blue
    ax.plot(outside_highlight_before.index, outside_highlight_before['Close'], label='Close Price (Outside Highlight)', color='blue')
    ax.plot(outside_highlight_after.index, outside_highlight_after['Close'], color='blue')

    # Plot highlighted region in orange
    ax.plot(within_highlight.index, within_highlight['Close'], label='Close Price (Highlight)', color='orange')

    # Add labels and legend
    ax.set_title("Stock Price with Highlighted Trend Line")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()

    # Display the plot in Streamlit
    st.pyplot(fig)

# --- FOOTER ---
import streamlit as st

# About Section
st.header("About")
st.write("""
I am **Nadir Ali Khan**, a student of the **PITP Certified Python Course** at IBA Sukkur 
and also enrolled in the **Certified Data Science Course**. Through these courses, I am 
honing my Python skills and diving deeper into data science, building projects that 
showcase my learning. This project reflects my progress and the knowledge I’ve gained 
so far. Thank you for exploring my work!
""")

# Skills Section
st.header("Skills")
st.write("""
- **Python**: Experienced with scripting, data analysis, and web development.
- **Machine Learning**: Basic understanding of ML algorithms.
- **Web Development**: Familiar with HTML, CSS, and JavaScript.
- **Data Analysis**: Proficient in using Pandas and Matplotlib for data manipulation and visualization.
""")

# Certifications Section
st.header("Certifications")
st.write("""
- **MERN Stack Certification**: Expertise in MongoDB, Express, React, and Node.js.
- **Cloud Computing Certification**: Knowledge of cloud platforms like AWS, Azure, Google Cloud.
- **Python Certified**: Completed an advanced course in Python, specializing in scripting, automation, data analysis, data visualization, and streamlined GUI development.
- **DIT (Diploma in Information Technology)**: Fundamental understanding of IT concepts and practices.

These certifications have equipped me with the skills necessary to work on a variety of technologies and platforms.
""")

# Client Reviews Section
st.header("Client Reviews")

st.subheader("Suhas (United States 🇺🇸)")
st.write("**Rating:** ⭐⭐⭐⭐⭐")
st.write("""
> *Great work! Nadir is very passionate about what he does, and very patient.  
  Will definitely work again in future — already have 3 new projects in queue.*
""")

st.subheader("Selvan (Malaysia 🇲🇾)")
st.write("**Rating:** ⭐⭐⭐⭐⭐")
st.write("""
> *Nadir is an excellent and very intelligent programmer.  
  He made a crypto sniping bot that works flawlessly.  
  If you need any bot-related program, look no further; Nadir is the guy!  
  A+++++ Highly recommended.*
""")

st.subheader("Zac Rosser (Australia 🇦🇺)")
st.write("**Rating:** ⭐⭐⭐⭐⭐")
st.write("""
> *Amazing work as always. Frequent communication and kept me up to date throughout the process.*
""")
st.write("**Seller communication level:** ⭐⭐⭐⭐⭐")


# Contact Section
st.header("Contact")
st.write("""
Feel free to reach out to me for any inquiries or collaboration opportunities:

- 📞 **Phone**: +92 304 2019543
- 📧 **Email**: [nadiralikhanofficial@gmail.com](mailto:nadiralikhanofficial@gmail.com)
""")

st.write("---")
st.markdown("""
<p style="text-align: center; font-size: 18px;">
    <a href="https://linkedin.com/in/teamnadiralikhan" target="_blank" style="text-decoration: none; color: #0A66C2; font-weight: bold; padding: 5px; border: 2px solid #0A66C2; border-radius: 5px; transition: background-color 0.3s, color 0.3s;">
        LinkedIn
    </a>
    &nbsp;&nbsp;|&nbsp;&nbsp;
    <a href="https://github.com/NadirAliOfficial" target="_blank" style="text-decoration: none; color: #333; font-weight: bold; padding: 5px; border: 2px solid #333; border-radius: 5px; transition: background-color 0.3s, color 0.3s;">
        GitHub
    </a>
</p>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; font-size: 14px; color: gray;">
    &copy; 2024 All rights reserved to Nadir Ali Khan
</div>
""", unsafe_allow_html=True)

