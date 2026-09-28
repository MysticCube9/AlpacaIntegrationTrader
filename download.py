import os
import yfinance as yf
import pandas as pd
import contextlib

# Current script directory
# Get the folder where the script is actually saved
script_dir = os.path.dirname(os.path.abspath(__file__))

# Create a subfolder "Stock_Data" inside the script folder
save_path = os.path.join(script_dir, "Stock_Data")

os.makedirs(save_path, exist_ok=True)

# List of stock symbols
symbols = [
    "MSFT", "NFLX", "AAPL", "TSLA", "INTC"
]

period = "max"

for symbol in symbols:
    print(f"Downloading {symbol}...")
    # Suppress verbose output
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
            stock_data = yf.download(symbol, period=period)

    # Flatten columns if multi-index exists
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = stock_data.columns.get_level_values(0)

    # Reset index so 'Date' becomes a column
    stock_data.reset_index(inplace=True)

    # Convert numeric columns to floats and round to 2 decimal places
    for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
        if col in stock_data.columns:
            stock_data[col] = pd.to_numeric(stock_data[col], errors='coerce').round(2)

    # Remove any extra rows that are not needed
    stock_data = stock_data[stock_data['Date'].notna()]

    # Save clean CSV
    csv_path = os.path.join(save_path, f"{symbol}.csv")
    stock_data.to_csv(csv_path, index=False)
    print(f"Saved {symbol} data ({len(stock_data)} rows) to {csv_path}")

print("All downloads completed.")
