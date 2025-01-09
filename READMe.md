

# Stock Price Analysis and Prediction

This repository contains a **Streamlit** application (`app.py`) that:
1. Fetches historical stock data (default: `AAPL`) from **Yahoo Finance**.  
2. Calculates technical indicators (SMA, RSI).  
3. Trains and/or loads a pre-trained **LSTM model** (via `lstm_stock_model.h5`) for future price predictions.  
4. Visualizes real-time plots, including a **highlighted period** plot.  
5. Showcases personal **skills, certifications**, and **client reviews** (bottom of the page).

---

## Features

- **Live Stock Data**: Pulls historical data using the [yfinance](https://pypi.org/project/yfinance/) library.  
- **Technical Indicators**: Automatically computes 50-day and 200-day Simple Moving Averages (SMA) and the Relative Strength Index (RSI).  
- **LSTM Forecast**: Loads a pre-trained LSTM model (`lstm_stock_model.h5`) to forecast user-defined future steps.  
- **Interactive Plots**: Displays various plots (e.g., close price, RSI, forecast) and allows highlighting specific date ranges.  
- **User Inputs**: Customize ticker, date range, and forecast steps via **Streamlit sidebar**.  
- **Personal Showcase**: Includes an "About" section, "Skills," "Certifications," "Client Reviews," and "Contact" details.

---

## Installation

1. **Clone** or **download** this repository:
   ```bash
   git clone https://github.com/YourUsername/your-repo-name.git
   cd your-repo-name
   ```

2. **Create and activate** a virtual environment (recommended):
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate

   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Make sure the following libraries are installed:
   - `streamlit`
   - `pandas`
   - `numpy`
   - `yfinance`
   - `matplotlib`
   - `tensorflow` or `tensorflow-cpu`
   - `scikit-learn`
   - and others as needed (listed in your `requirements.txt`)

4. (Optional) **Place your pre-trained LSTM model** in the same directory as `app.py` and name it `lstm_stock_model.h5` (or update the code if you use a different filename/path).

---

## Usage

1. **Start the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. **Open** the automatically generated URL (usually [http://localhost:8501](http://localhost:8501)) in your web browser.

3. **Interact** with the sidebar:
   - **Enter the Stock Ticker** (e.g., `AAPL`, `TSLA`, `MSFT`).  
   - **Pick the Start/End Dates** for historical data.  
   - **Adjust the Forecast Steps** to see how many days into the future you want predictions.  
   - Click **"Predict Future Prices"** to generate a forecast.  

4. **Explore**:
   - **Historical Data** and **DataFrame** preview.  
   - **Closing Price Plot** with 50-day & 200-day moving averages.  
   - **RSI Plot** with oversold/overbought lines (30 & 70).  
   - **Forecast Plot** showing predicted future prices.  
   - **Highlighted Period Plot** for a specific date range in the sidebar.  
   - **Personal/Portfolio Info** at the bottom of the page (About, Skills, Certifications, Client Reviews, etc.).

---

## Project Structure

```
your-repo-name/
├── app.py                  # Main Streamlit application
├── lstm_stock_model.h5     # (Optional) Pre-trained LSTM model
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## Customization

1. **Model Loading**: If you have your own LSTM model, rename or adjust the path in:
   ```python
   model_file = "lstm_stock_model.h5"
   ```
   Or specify a different path as needed.

2. **Date Range**: The default start/end dates can be changed in `st.sidebar.date_input`:
   ```python
   start_date = st.sidebar.date_input("Start Date", datetime.date(2015, 1, 1))
   end_date = st.sidebar.date_input("End Date", datetime.date.today())
   ```

3. **Technical Indicators**:  
   - Modify or add more indicators (e.g., MACD, Bollinger Bands) in `load_stock_data()`.

4. **Highlighted Period Plot**:  
   - Adjust the color, alpha, or style of the highlight in the last section of `app.py`.

---

## Known Issues / Troubleshooting

- **Empty DataFrame**: If no data is found for the selected ticker/date range, you may see “No data found.” Check your ticker symbol or date inputs.  
- **Model Not Found**: If `lstm_stock_model.h5` is missing or named differently, please rename or place your model in the same directory.  
- **Python/TensorFlow Versions**: If you face import or version conflicts, ensure your `requirements.txt` matches your local environment’s versions.

---

## Contributing

Contributions, suggestions, or bug reports are welcome! Feel free to:
1. **Open an Issue** for bug reports or feature requests.  
2. **Submit a Pull Request** with improvements or new features.

---

## License

You can include a license of your choice here (e.g., MIT, Apache 2.0) or declare **All Rights Reserved** if you prefer.

---

## Contact

- **Name**: [Nadir Ali Khan](mailto:nadiralikhanofficial@gmail.com)  
- **Email**: [nadiralikhanofficial@gmail.com](mailto:nadiralikhanofficial@gmail.com)  
- **LinkedIn**: [linkedin.com/in/teamnadiralikhan](https://linkedin.com/in/teamnadiralikhan)  
- **GitHub**: [github.com/NadirAliOfficial](https://github.com/NadirAliOfficial)

If you have any questions or would like to collaborate, feel free to reach out!

---

**Thank you for using the Stock Price Analysis and Prediction application!**