

import os
import sys
import math
import sqlite3
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATA_DIR = r"C:\Users\hp\OneDrive\Documents\Prince Gupta\Milestone-3\data3"  # <-- your data folder
BTC_FILE = "BTC-USD.csv"
ETH_FILE = "ETH-USD.csv"
DB_FILE = "portfolio.db"
PRED_CSV = "predicted_returns.csv"
PLOT_FILE = "returns_prediction.png"
FUTURE_DAYS = 5
WEIGHTS = np.array([0.5, 0.5])  # equal weights default

# ------------------------
# Helpers
# ------------------------
def find_file(name):
    """Look in working directory first, then DATA_DIR"""
    if os.path.exists(name):
        return name
    candidate = os.path.join(DATA_DIR, name)
    if os.path.exists(candidate):
        return candidate
    # try case-insensitive search in DATA_DIR
    if os.path.isdir(DATA_DIR):
        for f in os.listdir(DATA_DIR):
            if f.lower() == name.lower():
                return os.path.join(DATA_DIR, f)
    return None

def choose_price_column(df):
    """Return best price column name available in df"""
    for col in ["Adj Close", "Close", "close", "adj close"]:
        if col in df.columns:
            return col
    # fallback: try any column with numeric values and plausible name
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and "close" in col.lower():
            return col
    return None

def safe_read_csv(path):
    """Read CSV and print columns. Return DataFrame or raise."""
    print(f"Reading CSV: {path}")
    df = pd.read_csv(path)
    print(" Columns found:", df.columns.tolist())
    return df


USE_SKLEARN = True
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    print("sklearn available, will use LinearRegression.")
except Exception as e:
    USE_SKLEARN = False
    print("sklearn not available (will fallback to numpy.polyfit). Import error:", e)

# ------------------------
# Load data
# ------------------------
btc_path = find_file(BTC_FILE)
eth_path = find_file(ETH_FILE)

if not btc_path or not eth_path:
    print("ERROR: Could not find BTC or ETH CSV files.")
    print("Searched for:", BTC_FILE, "and", ETH_FILE)
    print("Looked in current dir:", os.getcwd())
    print("and DATA_DIR:", DATA_DIR)
    print("Files found in DATA_DIR:", os.listdir(DATA_DIR) if os.path.isdir(DATA_DIR) else "<no data dir>")
    sys.exit(1)

try:
    btc_df = safe_read_csv(btc_path)
    eth_df = safe_read_csv(eth_path)
except Exception as e:
    print("Failed to read CSV(s):", e)
    raise

# choose price column
btc_price_col = choose_price_column(btc_df)
eth_price_col = choose_price_column(eth_df)
if btc_price_col is None or eth_price_col is None:
    print("ERROR: Could not detect price column in one of CSVs. Columns were:")
    print("BTC columns:", btc_df.columns.tolist())
    print("ETH columns:", eth_df.columns.tolist())
    sys.exit(1)

print(f"Using BTC price column: '{btc_price_col}'")
print(f"Using ETH price column: '{eth_price_col}'")

# parse dates and prepare
for df, name in ((btc_df, "BTC"), (eth_df, "ETH")):
    if "Date" not in df.columns:
        print(f"ERROR: '{name}' file has no 'Date' column.")
        sys.exit(1)

btc_df['Date'] = pd.to_datetime(btc_df['Date'])
eth_df['Date'] = pd.to_datetime(eth_df['Date'])

# keep only Date + price columns
btc_small = btc_df[['Date', btc_price_col]].rename(columns={btc_price_col: 'Price_BTC'})
eth_small = eth_df[['Date', eth_price_col]].rename(columns={eth_price_col: 'Price_ETH'})

# merge on Date (inner join)
merged = pd.merge(btc_small, eth_small, on='Date', how='inner').sort_values('Date').reset_index(drop=True)
if merged.empty:
    print("ERROR: merged data is empty after joining on Date. Check overlapping date ranges.")
    sys.exit(1)

# calculate returns
merged['BTC_Return'] = merged['Price_BTC'].pct_change()
merged['ETH_Return'] = merged['Price_ETH'].pct_change()
merged = merged.dropna().reset_index(drop=True)
merged['Portfolio_Return'] = merged[['BTC_Return','ETH_Return']].dot(WEIGHTS)

print(f"Loaded {len(merged)} rows of aligned returns (after dropna). Date range {merged['Date'].iloc[0].date()} to {merged['Date'].iloc[-1].date()}")

# ------------------------
# Risk check (part 1)
# ------------------------
def compute_risk_metrics(port_series, weights):
    vol = port_series.std()
    rf = 0.01  # annual risk-free as placeholder (we used returns not annualized)
    # For daily returns rf should be converted; keep consistent with earlier approach:
    # Here we compute simple Sharpe on raw series (user can adjust)
    sharpe = (port_series.mean() - rf/252) / (port_series.std() if port_series.std() != 0 else np.nan)
    concentration = np.max(weights)
    return vol, sharpe, concentration

vol, sharpe, concentration = compute_risk_metrics(merged['Portfolio_Return'], WEIGHTS)
print(f"Risk metrics -> Volatility: {vol:.6f}   Sharpe: {sharpe:.6f}   Concentration: {concentration:.4f}")

# write risk metrics to DB
try:
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS risk_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        checked_at TEXT,
        metric TEXT,
        value REAL,
        status TEXT
    )
    """)
    # thresholds (example; you can adjust)
    alerts = []
    # Vol threshold
    vol_thresh = 0.05
    vol_status = 'PASS' if vol <= vol_thresh else 'FAIL'
    if vol_status == 'FAIL': alerts.append(f"Volatility {vol:.6f} > {vol_thresh}")
    cur.execute("INSERT INTO risk_metrics (checked_at, metric, value, status) VALUES (?, ?, ?, ?)",
                (datetime.utcnow().isoformat(), 'Volatility', float(vol), vol_status))
    # Sharpe threshold
    sharpe_thresh = 1.0
    sharpe_status = 'PASS' if (not math.isnan(sharpe) and sharpe >= sharpe_thresh) else 'FAIL'
    if sharpe_status == 'FAIL': alerts.append(f"Sharpe {sharpe:.6f} < {sharpe_thresh}")
    cur.execute("INSERT INTO risk_metrics (checked_at, metric, value, status) VALUES (?, ?, ?, ?)",
                (datetime.utcnow().isoformat(), 'Sharpe', float(sharpe) if not math.isnan(sharpe) else None, sharpe_status))
    # concentration threshold
    conc_thresh = 0.6
    conc_status = 'PASS' if concentration <= conc_thresh else 'FAIL'
    if conc_status == 'FAIL': alerts.append(f"Concentration {concentration:.4f} > {conc_thresh}")
    cur.execute("INSERT INTO risk_metrics (checked_at, metric, value, status) VALUES (?, ?, ?, ?)",
                (datetime.utcnow().isoformat(), 'Concentration', float(concentration), conc_status))

    conn.commit()
    conn.close()
    print("Risk metrics saved to DB (table: risk_metrics).")
except Exception as e:
    print("Error saving risk metrics to DB:", e)

# if alerts -> print and (optionally) send email
if alerts:
    print("ALERTS detected:")
    for a in alerts:
        print(" -", a)
    # Note: email sending lines are commented for safety; uncomment & configure to enable
    # try:
    #     msg = MIMEText("\n".join(alerts))
    #     msg['Subject'] = 'Portfolio Risk Alert'
    #     msg['From'] = 'youremail@example.com'
    #     msg['To'] = 'client@example.com'
    #     with smtplib.SMTP('smtp.gmail.com', 587) as s:
    #         s.starttls()
    #         s.login('youremail@example.com', 'app_password')
    #         s.send_message(msg)
    #     print("Email alert sent.")
    # except Exception as e:
    #     print("Failed to send email alert:", e)
else:
    print("No risk alerts (all PASS according to thresholds).")

# ------------------------
# Prediction (part 2)
# ------------------------
def fit_and_predict(series, n_steps=FUTURE_DAYS):
    y = series.values
    X = np.arange(len(y)).reshape(-1, 1)
    if USE_SKLEARN:
        try:
            model = LinearRegression()
            model.fit(X, y)
            future_X = np.arange(len(y), len(y)+n_steps).reshape(-1, 1)
            preds = model.predict(future_X)
            # compute training metrics
            train_pred = model.predict(X)
            mse = float(np.mean((train_pred - y) ** 2))
            r2 = float(1 - np.sum((y - train_pred)**2) / np.sum((y - np.mean(y))**2)) if np.var(y) != 0 else float('nan')
            return preds, mse, r2, 'sklearn'
        except Exception as e:
            print("sklearn LinearRegression failed, falling back to numpy.polyfit:", e)

    # fallback: numpy.polyfit (degree 1)
    coeffs = np.polyfit(X.flatten(), y, 1)
    # line: coeffs[0] * x + coeffs[1]
    future_x = np.arange(len(y), len(y)+n_steps)
    preds = coeffs[0] * future_x + coeffs[1]
    # training metrics (approx)
    train_pred = coeffs[0] * X.flatten() + coeffs[1]
    mse = float(np.mean((train_pred - y) ** 2))
    # r2 compute
    ss_res = np.sum((y - train_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else float('nan')
    return preds, mse, r2, 'polyfit'

# run predictions for BTC, ETH, Portfolio
btc_preds, btc_mse, btc_r2, btc_method = fit_and_predict(merged['BTC_Return'])
eth_preds, eth_mse, eth_r2, eth_method = fit_and_predict(merged['ETH_Return'])
port_preds, port_mse, port_r2, port_method = fit_and_predict(merged['Portfolio_Return'])

print(f"BTC prediction method: {btc_method}, train MSE={btc_mse:.6e}, R2={btc_r2:.4f}")
print(f"ETH prediction method: {eth_method}, train MSE={eth_mse:.6e}, R2={eth_r2:.4f}")
print(f"Portfolio prediction method: {port_method}, train MSE={port_mse:.6e}, R2={port_r2:.4f}")

# ------------------------
# Save predictions to CSV (combined with history)
# ------------------------
future_dates = pd.date_range(start=merged['Date'].iloc[-1] + pd.Timedelta(days=1), periods=FUTURE_DAYS)
hist = merged[['Date','BTC_Return','ETH_Return','Portfolio_Return']].copy()
hist = hist.rename(columns={'BTC_Return':'BTC_Actual','ETH_Return':'ETH_Actual','Portfolio_Return':'Portfolio_Actual'})

pred_df = pd.DataFrame({
    'Date': future_dates,
    'BTC_Predicted': btc_preds,
    'ETH_Predicted': eth_preds,
    'Portfolio_Predicted': port_preds
})

# Combine historic and predicted (predicted rows will have NaN in actuals)
combined = pd.concat([hist, pd.DataFrame({
    'Date': pred_df['Date'],
    'BTC_Actual': [np.nan]*len(pred_df),
    'ETH_Actual': [np.nan]*len(pred_df),
    'Portfolio_Actual': [np.nan]*len(pred_df),
    'BTC_Predicted': pred_df['BTC_Predicted'],
    'ETH_Predicted': pred_df['ETH_Predicted'],
    'Portfolio_Predicted': pred_df['Portfolio_Predicted'],
})], ignore_index=True)

combined.to_csv(PRED_CSV, index=False)
print(f"Predictions saved to CSV: {PRED_CSV}")

# also store only future predictions into DB table 'predictions'
try:
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT,
        asset TEXT,
        pred_day INTEGER,
        pred_date TEXT,
        predicted_return REAL,
        method TEXT
    )
    """)
    # insert future preds
    for i, (d, val) in enumerate(zip(pred_df['Date'], pred_df['BTC_Predicted']), start=1):
        cur.execute("INSERT INTO predictions (created_at, asset, pred_day, pred_date, predicted_return, method) VALUES (?, ?, ?, ?, ?, ?)",
                    (datetime.utcnow().isoformat(), 'BTC', i, str(d.date()), float(val), btc_method))
    for i, (d, val) in enumerate(zip(pred_df['Date'], pred_df['ETH_Predicted']), start=1):
        cur.execute("INSERT INTO predictions (created_at, asset, pred_day, pred_date, predicted_return, method) VALUES (?, ?, ?, ?, ?, ?)",
                    (datetime.utcnow().isoformat(), 'ETH', i, str(d.date()), float(val), eth_method))
    for i, (d, val) in enumerate(zip(pred_df['Date'], pred_df['Portfolio_Predicted']), start=1):
        cur.execute("INSERT INTO predictions (created_at, asset, pred_day, pred_date, predicted_return, method) VALUES (?, ?, ?, ?, ?, ?)",
                    (datetime.utcnow().isoformat(), 'Portfolio', i, str(d.date()), float(val), port_method))
    conn.commit()
    conn.close()
    print("Future predictions saved to DB table 'predictions'.")
except Exception as e:
    print("Error saving predictions to DB:", e)

# ------------------------
# Plot: historical + predicted
# ------------------------
plt.figure(figsize=(12,6))
plt.plot(merged['Date'], merged['BTC_Return'], label='BTC Actual', alpha=0.6)
plt.plot(merged['Date'], merged['ETH_Return'], label='ETH Actual', alpha=0.6)
plt.plot(merged['Date'], merged['Portfolio_Return'], label='Portfolio Actual', alpha=0.9, linewidth=2)

# predicted lines
plt.plot(pred_df['Date'], pred_df['BTC_Predicted'], '--', label='BTC Predicted')
plt.plot(pred_df['Date'], pred_df['ETH_Predicted'], '--', label='ETH Predicted')
plt.plot(pred_df['Date'], pred_df['Portfolio_Predicted'], '--', label='Portfolio Predicted')

plt.title('Actual vs Predicted Returns (next {} days)'.format(FUTURE_DAYS))
plt.xlabel('Date')
plt.ylabel('Return')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_FILE, dpi=300)
print(f"Chart saved to: {PLOT_FILE}")
plt.show()
