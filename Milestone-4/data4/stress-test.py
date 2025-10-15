# ---------- rule_setter_stress_test.py ----------
import pandas as pd
import numpy as np
import sqlite3

# ---------- LOAD & CLEAN DATA ----------
btc = pd.read_csv(r"C:\Users\hp\OneDrive\Documents\Prince Gupta\Milestone-3\data3\BTC-USD.csv")
eth = pd.read_csv(r"C:\Users\hp\OneDrive\Documents\Prince Gupta\Milestone-3\data3\ETH-USD.csv")

# Clean ETH 'Price' column (remove commas or symbols, convert to numeric)
eth['Price'] = eth['Price'].replace({',': ''}, regex=True)      # remove commas
eth['Price'] = pd.to_numeric(eth['Price'], errors='coerce')     # convert to float

# Ensure BTC 'Adj Close' is numeric
btc['Adj Close'] = pd.to_numeric(btc['Adj Close'], errors='coerce')
 
# Drop rows with invalid numeric values        
btc.dropna(subset=['Adj Close'], inplace=True)   
eth.dropna(subset=['Price'], inplace=True)

# Calculate daily returns
btc['Return'] = btc['Adj Close'].pct_change()
eth['Return'] = eth['Price'].pct_change()

btc = btc.dropna(subset=['Return'])
eth = eth.dropna(subset=['Return'])

print("✅ Data loaded and cleaned successfully!")

# ---------- RULE SETTER ----------
# Apply Risk-Parity Rule: weights inversely proportional to volatility
btc_vol = btc['Return'].std()
eth_vol = eth['Return'].std()

inv_vol_btc = 1 / btc_vol
inv_vol_eth = 1 / eth_vol
weights = np.array([inv_vol_btc, inv_vol_eth]) / (inv_vol_btc + inv_vol_eth)

print(f"📊 Portfolio Weights (Risk-Parity) -> BTC: {weights[0]:.2f}, ETH: {weights[1]:.2f}")

# Calculate Portfolio Returns
portfolio_returns = (weights[0] * btc['Return']) + (weights[1] * eth['Return'])

# ---------- STORE IN DATABASE ----------
conn = sqlite3.connect("rule_stress_test.db")
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS portfolio_results(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rule TEXT,
    btc_weight REAL,
    eth_weight REAL,
    avg_return REAL,
    volatility REAL
)
""")
conn.commit()

avg_return = portfolio_returns.mean()
portfolio_volatility = portfolio_returns.std()

c.execute("INSERT INTO portfolio_results(rule, btc_weight, eth_weight, avg_return, volatility) VALUES (?, ?, ?, ?, ?)",
          ("Risk-Parity", weights[0], weights[1], avg_return, portfolio_volatility))
conn.commit()

print("✅ Portfolio returns stored in rule_stress_test.db")

# ---------- STRESS TEST ----------
# Apply a 20% market shock
shock_factor = 0.2
btc_shocked = btc['Adj Close'] * (1 - shock_factor)
eth_shocked = eth['Price'] * (1 - shock_factor)

btc['Shocked_Return'] = btc_shocked.pct_change()
eth['Shocked_Return'] = eth_shocked.pct_change()

portfolio_shocked_returns = (weights[0] * btc['Shocked_Return']) + (weights[1] * eth['Shocked_Return'])

shocked_avg_return = portfolio_shocked_returns.mean()
shocked_volatility = portfolio_shocked_returns.std()

c.execute("INSERT INTO portfolio_results(rule, btc_weight, eth_weight, avg_return, volatility) VALUES (?, ?, ?, ?, ?)",
          ("Stress-Test (20% Drop)", weights[0], weights[1], shocked_avg_return, shocked_volatility))
conn.commit()
conn.close()

print("✅ Stress test applied and results saved!")

# ---------- OUTPUT ----------
print("\n--- RESULTS ---")
print(f"Normal Avg Return: {avg_return:.4f}, Volatility: {portfolio_volatility:.4f}")
print(f"Stress-Test Avg Return: {shocked_avg_return:.4f}, Volatility: {shocked_volatility:.4f}")
