# Stock Market Analysis & Prediction

A Python project that analyzes historical stock data and predicts future stock prices using LSTM (Long Short-Term Memory) neural networks.

## Features
- Download and analyze stock data using yfinance API
- Visualize stock trends, daily returns, volatility, and volume
- Predict future stock prices using LSTM model
- Compare predicted vs actual stock prices

## Dependencies

```bash
pip install tensorflow-macos
pip install tensorflow-metal
pip install yfinance pandas numpy matplotlib seaborn scikit-learn
```

## Project Structure

The project consists of several key functions:

1. `get_stock_data(ticker, start_date, end_date)`: Downloads historical stock data using yfinance API
2. `prepare_data(df, lookback=60)`: Preprocesses data for LSTM model including:
   - Data scaling
   - Sequence creation
   - Train/test splitting
3. `create_model(lookback)`: Creates LSTM neural network with:
   - Two LSTM layers (50 units each)
   - Dropout layers for regularization
   - Final Dense layer for prediction
4. `plot_stock_analysis(df, ticker)`: Generates visualization plots for:
   - Stock price trends
   - Daily returns distribution
   - Volatility patterns
   - Trading volume

## Usage

```python
# Example usage
ticker = "AAPL"  # Stock symbol (e.g., AAPL for Apple)
start_date = "2020-01-01"
end_date = "2023-12-31"
lookback = 60  # Number of days to look back for prediction

# Run the analysis
python stock_analysis.py
```

## Output

The script generates multiple visualizations:
1. Stock analysis plots:
   - Historical price trends
   - Distribution of daily returns
   - 30-day rolling volatility
   - Trading volume over time
2. Prediction results:
   - Actual vs predicted stock prices

## Setup & Installation

1. Create and activate Conda environment:
```bash
conda create -n stockenv python=3.9
conda activate stockenv
```

2. Install required packages:
```bash
conda install tensorflow-macos tensorflow-metal
conda install yfinance pandas numpy matplotlib seaborn scikit-learn
```

## Running the Project

1. Activate Conda environment (if not already activated):
```bash
conda activate stockenv
```

2. Run the analysis:
```bash
python stock_analysis.py
```

This will generate visualizations for stock analysis and price predictions using default parameters (AAPL stock, 2020-2023 data).
