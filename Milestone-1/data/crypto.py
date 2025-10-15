# ---------- Portfolio_math.py (Enhanced with Auto File Detection) ----------
import os
import pandas as pd
import numpy as np
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import csv

# ---------- AUTO-DETECT CSV FILES ----------
def find_file(filename, search_path="."):
    for root, dirs, files in os.walk(search_path):
        if filename in files:
            return os.path.join(root, filename)
    return None

btc_path = find_file("BTC-USD.csv", os.getcwd())
eth_path = find_file("ETH-USD.csv", os.getcwd())

if not btc_path or not eth_path:
    print("❌ Error: One or both CSV files not found. Please ensure BTC-USD.csv and ETH-USD.csv exist.")
    print("🔍 Current directory:", os.getcwd())
    exit()

# ---------- LOAD DATA ----------
btc = pd.read_csv(btc_path)
eth = pd.read_csv(eth_path)

print(f"✅ Loaded BTC data from: {btc_path}")
print(f"✅ Loaded ETH data from: {eth_path}")

# ---------- CALCULATE EXPECTED RETURNS ----------
btc['Return'] = btc['Close'].pct_change()
eth['Return'] = eth['Close'].pct_change()

expected_returns = np.array([btc['Return'].mean(), eth['Return'].mean()])
cov_matrix = np.cov(btc['Return'].dropna(), eth['Return'].dropna())
currencies = ['BTC', 'ETH']

# ---------- PORTFOLIO CALCULATIONS ----------
def calculate_portfolio(weights):
    port_return = np.dot(weights, expected_returns)
    port_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    port_risk = np.sqrt(port_variance)
    return port_return, port_risk

def equal_weight_rule():
    weights = np.array([0.5, 0.5])
    return calculate_portfolio(weights), weights

def risk_based_rule():
    asset_risks = np.sqrt(np.diag(cov_matrix))
    inv_risks = 1 / asset_risks
    weights = inv_risks / np.sum(inv_risks)
    return calculate_portfolio(weights), weights

def performance_based_rule():
    weights = expected_returns / np.sum(expected_returns)
    return calculate_portfolio(weights), weights

# ---------- STORE IN DATABASE ----------
def run_rule_and_store(rule_name, strategy_func):
    (port_return, port_risk), weights = strategy_func()

    conn = sqlite3.connect('portfolio.db')
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO portfolio (rule_name, total_return, total_risk)
        VALUES (?, ?, ?)
    ''', (rule_name, port_return, port_risk))
    portfolio_id = cursor.lastrowid

    for i in range(len(currencies)):
        asset_variance = cov_matrix[i][i]
        asset_risk = np.sqrt(asset_variance)
        cursor.execute('''
            INSERT INTO portfolio_assets (portfolio_id, currency, weight, expected_return, asset_risk)
            VALUES (?, ?, ?, ?, ?)
        ''', (portfolio_id, currencies[i], weights[i], expected_returns[i], asset_risk))

    conn.commit()
    conn.close()
    print(f"[{rule_name}] Stored in DB.")

# ---------- CREATE TABLES IF NOT EXISTS ----------
conn = sqlite3.connect('portfolio.db')
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS portfolio (
        portfolio_id INTEGER PRIMARY KEY AUTOINCREMENT,
        rule_name TEXT,
        total_return REAL,
        total_risk REAL
    )
''')
cursor.execute('''
    CREATE TABLE IF NOT EXISTS portfolio_assets (
        asset_id INTEGER PRIMARY KEY AUTOINCREMENT,
        portfolio_id INTEGER,
        currency TEXT,
        weight REAL,
        expected_return REAL,
        asset_risk REAL,
        FOREIGN KEY (portfolio_id) REFERENCES portfolio (portfolio_id)
    )
''')
conn.commit()
conn.close()

# ---------- RUN STRATEGIES IN PARALLEL ----------
strategies = {
    'Equal Weight': equal_weight_rule,
    'Risk-Based': risk_based_rule,
    'Performance-Based': performance_based_rule
}

with ThreadPoolExecutor() as executor:
    futures = [executor.submit(run_rule_and_store, name, func)
               for name, func in strategies.items()]

# ---------- COMPARE PORTFOLIO VS SINGLE ASSETS ----------
conn = sqlite3.connect('portfolio.db')
cursor = conn.cursor()
cursor.execute('SELECT rule_name, total_return FROM portfolio')
strategy_data = cursor.fetchall()
conn.close()

single_assets = [('BTC', expected_returns[0]), ('ETH', expected_returns[1])]
comparison_data = strategy_data + single_assets

# ---------- LINE CHART ----------
labels = [row[0] for row in comparison_data]
returns = [row[1] for row in comparison_data]

plt.figure(figsize=(10, 6))
plt.plot(labels, returns, marker='o', linewidth=2.5, color='blue')
plt.title('Portfolio Return vs Single Asset Return', fontsize=14)
plt.ylabel('Expected Return')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# ---------- EXPORT TO CSV ----------
with open('portfolio_vs_assets.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Name', 'Return'])
    writer.writerows(comparison_data)

print("\n✅ Comparison data exported to 'portfolio_vs_assets.csv'")
print("\n--- Comparison Data ---")
for row in comparison_data:
    print(f"{row[0]}: {row[1]:.4f}")
