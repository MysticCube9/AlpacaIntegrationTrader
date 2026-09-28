# Gregory Tishkevich - Advanced Programming
# TSLA Stock Prediction: Gradient Boosting + LSTM with walk-forward backtesting

import os
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf


# ----------- Helper functions -----------

def load_data(path):
    """Load CSV and preprocess it."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Stock historical CSV not found at {path}")
    df = pd.read_csv(path)
    return preprocess(df)


def preprocess(df):
    """Clean stock dataframe: remove commas, convert numeric, add previous day features."""
    df = df.copy()
    for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    df['Prev_High'] = df['High'].shift(1).fillna(df['High'])
    df['Prev_Low'] = df['Low'].shift(1).fillna(df['Low'])
    df['Prev_Volume'] = df['Volume'].shift(1).fillna(df['Volume'])

    return df


def prepare_features(df, feature_cols=None, target_col='Close'):
    """Return X (features) and y (target)."""
    if feature_cols is None:
        feature_cols = ['Open', 'Prev_High', 'Prev_Low', 'Prev_Volume']
    X = df[feature_cols]
    y = df[target_col]
    return X, y


def impute_features(X, strategy='mean'):
    """Impute missing values."""
    imputer = SimpleImputer(strategy=strategy)
    X_imputed = imputer.fit_transform(X)
    return X_imputed, imputer


def train_gradient_boosting(X_train, y_train, n_estimators=500, max_depth=5, learning_rate=0.05, sample_weight=None):
    """Train Gradient Boosting model."""
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model on test set."""
    y_pred = model.predict(X_test)
    r2 = metrics.r2_score(y_test, y_pred)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    return r2, mae, y_pred


def apply_recency_bias(df, X_imputed, y, days=60, recent_weight=5):
    """Apply weights to give recent data more importance."""
    weights = np.ones(len(X_imputed))
    cutoff_date = df['Date'].max() - pd.Timedelta(days=days)
    recent_mask = df['Date'] >= cutoff_date
    weights[recent_mask] = recent_weight
    return weights


def get_latest_price(symbol, API_KEY, SECRET_KEY):
    """Fetch the most recent trade price from Alpaca."""
    url = f"https://data.alpaca.markets/v2/stocks/{symbol}/trades/latest"
    headers = {"APCA-API-KEY-ID": API_KEY, "APCA-API-SECRET-KEY": SECRET_KEY}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()['trade']['p']


def predict_today(model, imputer, df, open_price):
    """Predict today's closing price and return a trade signal."""
    if df.shape[0] < 2:
        print("Not enough data to make prediction")
        return "HOLD"

    yesterday_row = df.iloc[-1]

    today_features = pd.DataFrame({
        'Open': [open_price],
        'Prev_High': [yesterday_row['High']],
        'Prev_Low': [yesterday_row['Low']],
        'Prev_Volume': [yesterday_row['Volume']]
    })

    today_features_imputed = imputer.transform(today_features)
    today_pred = model.predict(today_features_imputed)[0]

    if today_pred > open_price:
        action = "BUY"
    elif today_pred < open_price:
        action = "SELL"
    else:
        action = "HOLD"

    print(f"Predicted closing price: ${today_pred:.2f}, Action: {action}")
    return action


# ----------- Gradient Boosting backtest -----------

def predict_last_days(days):
    """Backtest the gradient boosting model over the last `days` trading days."""
    dictionary = {}

    csv_folder = r"C:\Users\gtish\OneDrive - Eastside Preparatory School\Predict\Stock_Data"
    csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]

    target_symbol = "TSLA"
    file = f"{target_symbol}.csv"

    if file in csv_files:
        path = os.path.join(csv_folder, file)
    else:
        print(f"{file} not found in folder")
        return {}

    full_df = pd.read_csv(path)
    full_df = preprocess(full_df)

    if full_df.shape[0] < 2:
        print(f"{target_symbol}: Not enough data to train.")
        return {}

    feature_cols = ['Open', 'Prev_High', 'Prev_Low', 'Prev_Volume']
    total_rows = len(full_df)
    start_idx = total_rows - days

    if start_idx < 1:
        print("Not enough historical data before the requested backtest window.")
        return {}

    for i in range(days):
        target_idx = start_idx + i
        train_df = full_df.iloc[:target_idx]

        if train_df.shape[0] < 2:
            continue

        X, y = prepare_features(train_df)
        X_imputed, imputer = impute_features(X)
        weights = apply_recency_bias(train_df, X_imputed, y)
        model = train_gradient_boosting(X_imputed, y, sample_weight=weights)

        target_row = full_df.iloc[target_idx]
        open_price = target_row['Open']
        date = target_row['Date']

        signal = predict_today(model, imputer, train_df, open_price)
        dictionary[date] = signal
        print(f"{date}: open=${open_price:.2f}, signal={signal}")

    return dictionary


# ----------- LSTM helpers -----------

def prepare_lstm_sequences(df, feature_cols, target_col='Close', lookback=20):
    """
    Build (X, y) sequences for LSTM.
    X shape: (samples, lookback, features)
    y shape: (samples,)
    """
    data = df[feature_cols + [target_col]].values
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i, :-1])
        y.append(data[i, -1])
    return np.array(X), np.array(y)


def build_lstm_model(input_shape):
    """Build and compile a small LSTM model."""
    tf.function(reduce_retracing=True)
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# ----------- LSTM backtest -----------

def predict_last_days_lstm(days, lookback=20, walk_forward=True):
    """
    Backtest an LSTM model over the last `days` trading days.
    walk_forward=True  — train once then update incrementally each day (fast)
    walk_forward=False — retrain from scratch each day (slow but fully independent)
    """
    dictionary = {}

    csv_folder = r"C:\Users\gtish\OneDrive - Eastside Preparatory School\Predict\Stock_Data"
    csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]

    target_symbol = "TSLA"
    file = f"{target_symbol}.csv"

    if file in csv_files:
        path = os.path.join(csv_folder, file)
    else:
        print(f"{file} not found in folder")
        return {}

    full_df = pd.read_csv(path)
    full_df = preprocess(full_df)

    if full_df.shape[0] < lookback + days + 1:
        print("Not enough historical data for the requested lookback + backtest window.")
        return {}

    feature_cols = ['Open', 'Prev_High', 'Prev_Low', 'Prev_Volume']
    total_rows = len(full_df)
    start_idx = total_rows - days

    if start_idx < lookback + 1:
        print("Not enough historical data before the requested backtest window.")
        return {}

    # ----------------------------------------------------------------
    # Walk-forward: train once on everything before the backtest window
    # ----------------------------------------------------------------
    if walk_forward:
        train_df = full_df.iloc[:start_idx].copy()

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        train_features = scaler_X.fit_transform(train_df[feature_cols])
        train_target = scaler_y.fit_transform(train_df[['Close']])

        scaled_df = pd.DataFrame(
            np.hstack([train_features, train_target]),
            columns=feature_cols + ['Close']
        )

        X_train, y_train = prepare_lstm_sequences(scaled_df, feature_cols, lookback=lookback)

        model = build_lstm_model(input_shape=(lookback, len(feature_cols)))
        early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0, callbacks=[early_stop])

    # ----------------------------------------------------------------
    # Loop over each day in the backtest window
    # ----------------------------------------------------------------
    for i in range(days):
        target_idx = start_idx + i

        if target_idx - lookback < 0:
            continue

        # --- Retrain mode: build and fit a fresh model each iteration ---
        if not walk_forward:
            train_df = full_df.iloc[:target_idx].copy()

            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()

            train_features = scaler_X.fit_transform(train_df[feature_cols])
            train_target = scaler_y.fit_transform(train_df[['Close']])

            scaled_df = pd.DataFrame(
                np.hstack([train_features, train_target]),
                columns=feature_cols + ['Close']
            )

            X_train, y_train = prepare_lstm_sequences(scaled_df, feature_cols, lookback=lookback)

            if len(X_train) == 0:
                continue

            model = build_lstm_model(input_shape=(lookback, len(feature_cols)))
            early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
            model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0, callbacks=[early_stop])

        # --- Predict ---
        target_row = full_df.iloc[target_idx]
        open_price = target_row['Open']
        date = target_row['Date']

        sequence_df = full_df.iloc[target_idx - lookback:target_idx][feature_cols]
        sequence_scaled = scaler_X.transform(sequence_df)
        sequence_input = sequence_scaled.reshape(1, lookback, len(feature_cols))

        pred_scaled = model.predict(sequence_input, verbose=0)[0][0]
        predicted_close = scaler_y.inverse_transform([[pred_scaled]])[0][0]

        if predicted_close > open_price:
            signal = "BUY"
        elif predicted_close < open_price:
            signal = "SELL"
        else:
            signal = "HOLD"

        print(f"{date}: open=${open_price:.2f}, predicted close=${predicted_close:.2f}, signal={signal}")
        dictionary[date] = signal

        # --- Walk-forward update: a few epochs on the new day ---
        if walk_forward:
            update_window = full_df.iloc[target_idx - lookback:target_idx + 1].copy()

            update_features = scaler_X.transform(update_window[feature_cols])
            update_target = scaler_y.transform(update_window[['Close']])

            update_scaled_df = pd.DataFrame(
                np.hstack([update_features, update_target]),
                columns=feature_cols + ['Close']
            )

            X_new, y_new = prepare_lstm_sequences(update_scaled_df, feature_cols, lookback=lookback)

            if len(X_new) > 0:
                model.fit(X_new, y_new, epochs=5, batch_size=1, verbose=0)

    return dictionary


# ----------- Accuracy calculation -----------

def prediction_accuracy_calculation(dictionary, investment):
    """Evaluate signal dictionary against actual price movements."""
    csv_folder = r"C:\Users\gtish\OneDrive - Eastside Preparatory School\Predict\Stock_Data"
    csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]

    target_symbol = "TSLA"
    file = f"{target_symbol}.csv"

    if file in csv_files:
        path = os.path.join(csv_folder, file)
    else:
        print(f"{file} not found in folder")
        return

    print(dictionary)

    df = pd.read_csv(path)
    df = preprocess(df)

    running_total = investment
    correct_predictions = 0
    total_days = 0

    for date, signal in dictionary.items():
        date = pd.to_datetime(date)
        match = df[df['Date'] == date]

        if match.empty:
            print(f"Date {date} not found in CSV, skipping.")
            continue

        row = match.iloc[0]
        open_price = row['Open']
        close_price = row['Close']

        if open_price == 0:
            continue

        percent_change = (close_price - open_price) / open_price
        total_days += 1

        if signal == "BUY":
            running_total *= (1 + percent_change)
            if percent_change > 0:
                correct_predictions += 1

        elif signal == "SELL":
            running_total *= (1 - percent_change)
            if percent_change < 0:
                correct_predictions += 1

        else:  # HOLD
            running_total *= (1 + percent_change)
            if abs(percent_change) <= 0.003:
                correct_predictions += 1

    if total_days == 0:
        print("No matching dates found.")
        return

    accuracy = correct_predictions / total_days
    print(f"The model predicted {correct_predictions} correct out of {total_days}, accuracy: {accuracy:.2%}")
    print(f"Initial investment: ${investment:,.2f} → Final value: ${running_total:,.2f}")


# ----------- Entry point -----------

# Gradient Boosting backtest
# dictionary_gb = predict_last_days(days=30)
# prediction_accuracy_calculation(dictionary_gb, 100000)

# LSTM backtest (walk-forward)
dictionary_lstm = predict_last_days_lstm(days=30, lookback=20, walk_forward=False)
prediction_accuracy_calculation(dictionary_lstm, 100000)