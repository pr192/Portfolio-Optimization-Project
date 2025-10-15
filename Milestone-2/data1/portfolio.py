import sqlite3
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import csv


currencies = ['BTC', 'ETH']
expected_returns = np.array([0.12, 0.18])
cov_matrix = np.array([
    [0.005, -0.010],
    [-0.010, 0.040]
]) 

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

# Create Tables
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


strategies = {
    'equal_weight': equal_weight_rule,
    'risk_based': risk_based_rule,
    'performance_based': performance_based_rule
}

with ThreadPoolExecutor() as executor:
    futures = [executor.submit(run_rule_and_store, name, func)
               for name, func in strategies.items()]

# Compare Portfolio vs Single Asset Returns
conn = sqlite3.connect('portfolio.db')
cursor = conn.cursor()

cursor.execute('SELECT rule_name, total_return FROM portfolio')
strategy_data = cursor.fetchall()

single_assets = [('BTC', 0.12), ('ETH', 0.18)]
comparison_data = strategy_data + single_assets

# Line Graph 
labels = [row[0] for row in comparison_data]
returns = [row[1] for row in comparison_data]

plt.figure(figsize=(12, 7))
plt.plot(labels, returns, marker='o', linestyle='-', linewidth=2, markersize=8, color='teal')
plt.grid(visible=True, linestyle='--', linewidth=0.5, alpha=0.7)

for i, value in enumerate(returns):
    plt.text(i, value + 0.005, f'{value:.3f}', ha='center', fontsize=10)

plt.title('Portfolio vs Single Asset Return Comparison', fontsize=16, fontweight='bold')
plt.xlabel('Strategy / Asset', fontsize=14)
plt.ylabel('Return', fontsize=14)
plt.xticks(rotation=25, fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()

# Export to CSV
with open('portfolio_vs_assets.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Name', 'Return'])
    writer.writerows(comparison_data)

print("\n Comparison data exported to 'portfolio_vs_assets.csv'")


print("\n--- Comparison Data ---")
for row in comparison_data:
    print(f"{row[0]}: {row[1]:.4f}")

conn.close()
