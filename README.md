# Stock Price Prediction

Machine learning model for stock price prediction using LSTM neural networks on historical OHLCV data.

## Features
- LSTM-based time series forecasting
- Data preprocessing and normalization
- Visualization of predictions vs actual prices
- Configurable lookback window and epochs

## Requirements
```
pip install pandas numpy matplotlib scikit-learn tensorflow yfinance
```

## Usage
```bash
python predict.py --ticker AAPL --days 30
```

## Model Architecture
- Input: 60-day OHLCV window
- 2x LSTM layers with dropout
- Dense output layer for next-day close prediction

## License
MIT
<!-- updated: 2024-02-22-r01 -->
