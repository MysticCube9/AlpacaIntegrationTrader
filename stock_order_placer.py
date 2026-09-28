import os
import time
import pandas as pd
import pytz
import requests
import schedule
from datetime import datetime, timedelta
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import OrderStatus

# Import your TSLA predictor functions
from stock_predict import (
    load_data,
    preprocess,
    prepare_features,
    impute_features,
    train_gradient_boosting,
    apply_recency_bias,
    predict_today
)

# ------------------------------
# Alpaca API setup (Paper Trading)
# ------------------------------
API_KEY = "put key here"
SECRET_KEY = "put key here"
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

trained_models = {}  # will hold {symbol: (model, imputer, df)}

# ------------------------------
# Config
# ------------------------------
# folder to where you save your stock data
csv_folder = r"C:\Users\user\###"

percent_per_stock = 0.046  # allocate 4.6% of account per stock
allocation_ratio = 0.7   # invest only 70% of balance

# time config
eastern = pytz.timezone("US/Eastern")
now_est = datetime.now(eastern)

# Get today's market open in EST and convert to UTC
market_open_est = eastern.localize(datetime(now_est.year, now_est.month, now_est.day, 9, 30))
market_open_utc = market_open_est.astimezone(pytz.UTC)
market_close_utc = market_open_utc + timedelta(minutes=1)

start = market_open_utc.isoformat()
end = market_close_utc.isoformat()
#budget_per_stock = 100  # how much to allocate per stock

headers = {
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": SECRET_KEY
}

url = "https://data.alpaca.markets/v2/stocks/MSFT/bars?timeframe=1Min&limit=1"
r = requests.get(url, headers=headers)
print(r.status_code, r.text)


symbol_map = {
    "BRK-B": "BRK.B",
    "BRK-A": "BRK.A"
    # add others if needed
}

# ------------------------------
# Helper functions
# ------------------------------
def market_is_open():
    """Check if market is currently open."""
    try:
        clock = trading_client.get_clock()
        return clock.is_open
    except Exception as e:
        print(f"[{datetime.now()}] Error fetching market clock: {e}")
        return False     

def download_stock_data():

    # get most recent data
    # make this a reference to download.py
    os.system('python "C:\\Users\\user\\###\\download.py"')

    last_file = os.path.join(csv_folder, "GOOG.csv")

    # Wait until all files exist
    while True:
        if os.path.exists(last_file):
            # get last modification time
            mtime = os.path.getmtime(last_file)
            file_date = datetime.fromtimestamp(mtime).date()
            today = datetime.now().date()

            if file_date == today:
                print("Google file was updated today. Proceeding to trading...")
                break
            else:
                print(f"Google file exists but last modified on {file_date}, waiting for today’s update...")
        else:
            print("Waiting for GOOG.csv to appear...")

        time.sleep(10)  # check every 10 seconds
    
def train_models():

    """Train the models before the market opens to be ready when market opens."""
    print(f"[{datetime.now()}] Training models...")
    
    csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]
            
    for file in csv_files:
        symbol = file.replace('.csv', '')
        path = os.path.join(csv_folder, file)
        
        # --- Load and preprocess ---
        try:
            df = pd.read_csv(path)
            from stock_RandomForest import preprocess  # ensure imported
            df = preprocess(df)
        except Exception as e:
            print(f"{symbol}: Failed to load/preprocess data. Error: {e}")
            continue
        
        if df.empty or df.shape[0] < 2:
            print(f"{symbol}: Not enough data to make a prediction. Skipping.")
            continue
        
        # --- Prepare features ---
        from stock_RandomForest import prepare_features, impute_features, train_gradient_boosting, apply_recency_bias, predict_today, evaluate_model
        X, y = prepare_features(df)
        X_imputed, imputer = impute_features(X)
        
        # Train/test split (optional, last 30 days for evaluation)
        split_index = -30
        
        #X_train, y_train = X_imputed[:split_index], y[:split_index]

        X_train, X_test = X_imputed[:split_index], X_imputed[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        
        model = train_gradient_boosting(X_train, y_train)

        r2, mae, y_pred_test = evaluate_model(model, X_test, y_test)
        print(f"{symbol} evaluation → R²: {r2:.4f}, MAE: {mae:.2f}")

        
        # Train model and apply recency bias
        weights = apply_recency_bias(df, X_imputed, y)
        model = train_gradient_boosting(X_train, y_train)
        model.fit(X_imputed, y, sample_weight=weights)

        trained_models[symbol] = (model, imputer, df)

        
#----------------------------------------------------------------------------------
def trade_at_open():

    if not market_is_open():
        print(f"[{datetime.now()}] Market closed, skipping trading.")
        return
    
    print(f"[{datetime.now()}] Running predictions and placing orders...")

    csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]

    
    # Get account info once
    try:
        account = trading_client.get_account()
        account_value = float(account.cash)
    except Exception as e:
        print(f"Failed to fetch account info: {e}")
        return

    for symbol, (model, imputer, df) in trained_models.items():
        
        alpaca_symbol = symbol_map.get(symbol, symbol)

        try:
            # Get today's 9:30 AM Eastern market open
            now_est = datetime.now(eastern)

            market_open_est = eastern.localize(
                datetime(
                    now_est.year,
                    now_est.month,
                    now_est.day,
                    9,
                    30
                )
            )

            market_close_est = market_open_est + timedelta(minutes=1)

            # Convert Eastern time to UTC for Alpaca
            market_open_utc = market_open_est.astimezone(pytz.UTC)
            market_close_utc = market_close_est.astimezone(pytz.UTC)

            start = market_open_utc.isoformat()
            end = market_close_utc.isoformat()

            url = (
                f"https://data.alpaca.markets/v2/stocks/{alpaca_symbol}/bars"
                f"?timeframe=1Min&start={start}&end={end}"
            )

            headers = {
                "APCA-API-KEY-ID": API_KEY,
                "APCA-API-SECRET-KEY": SECRET_KEY
            }

            response = requests.get(url, headers=headers)
            response.raise_for_status()

            bars = response.json()["bars"]

            if not bars:
                print(f"{symbol}: No opening bar available yet.")
                continue

            bar = bars[0]
            open_price = bar["o"]


            
        except Exception as e:
            print(f"{symbol}: Failed to fetch open. Error: {e}")

        signal = predict_today(model, imputer, df, open_price)
        if signal == "HOLD":
            print(f"{symbol}: HOLD signal, no order placed.")
            continue      

        
        
        # --- Fetch latest price ---
        try:
            url = f"https://data.alpaca.markets/v2/stocks/{alpaca_symbol}/trades/latest"
            headers = {"APCA-API-KEY-ID": API_KEY, "APCA-API-SECRET-KEY": SECRET_KEY}
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            latest_price = response.json()['trade']['p']
        except Exception as e:
            print(f"{alpaca_symbol}: Failed to fetch latest price. Error: {e}")
            continue

        # --- Calculate quantity based on account percentage ---

        total_funds = account_value * allocation_ratio

        main_symbol = "TSLA"
        secondary_symbol = "NFLX"

        main_allocation = total_funds * 0.5
        secondary_allocation = total_funds * 0.2

        # other symbols split the rest
        csv_symbols = [f.replace('.csv', '') for f in csv_files]
        other_symbols = [s for s in csv_symbols if s not in [main_symbol, secondary_symbol]]
        other_allocation = (total_funds - main_allocation - secondary_allocation) / len(other_symbols)


        if symbol == main_symbol:
            budget = main_allocation
        elif symbol == secondary_symbol:
            budget = secondary_allocation
        else:
            budget = other_allocation

        #budget = allocations.get(symbol, 0)
        qty = max(int(budget / latest_price), 1)
            
        side = OrderSide.BUY if signal == "BUY" else OrderSide.SELL
        order_data = MarketOrderRequest(
            symbol=alpaca_symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY
        )
          
        # --- Submit order ---
        try:
            order = trading_client.submit_order(order_data=order_data)
            action = "long" if side == OrderSide.BUY else "short"
            print(f"{symbol}: {signal} {qty} @ ~${latest_price:.2f} ({action}) (Order ID: {order.id})")
        except Exception as e:
            print(f"{symbol}: Failed to place order. Error: {e}")
        
    print("Finished placing trades for the day.")


def liquidate_positions():
    """Sell everything before close to avoid overnight positions."""
    if not market_is_open():
        print(f"[{datetime.now()}] Market closed, skipping liquidation.")
        return
    
    print(f"[{datetime.now()}] Liquidating all positions...")
    positions = trading_client.get_all_positions()

    for position in positions:
        symbol = position.symbol
        alpaca_symbol = symbol_map.get(symbol, symbol)
        qty = abs(int(float(position.qty)))
        if qty > 0:
            side = OrderSide.SELL if float(position.qty) > 0 else OrderSide.BUY
            order_data = MarketOrderRequest(
                symbol=alpaca_symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY
            )
            try:
                trading_client.submit_order(order_data=order_data)
                print(f"Closed {qty} shares of {symbol}.")
            except Exception as e:
                print(f"Failed to close {symbol}: {e}")

    print("All positions liquidated.")


# ------------------------------
# Schedule tasks
# ------------------------------

schedule.every().day.at("06:15").do(train_models)               # pre-train models
schedule.every().day.at("06:30").do(trade_at_open)              # execute the trades
schedule.every().day.at("12:59").do(liquidate_positions)        # liquidate before close
schedule.every().day.at("13:25").do(download_stock_data)        # download and run

print("Scheduler started. Waiting for market hours...")


while True:
    schedule.run_pending()
    time.sleep(60)