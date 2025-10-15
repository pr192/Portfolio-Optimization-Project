import sqlite3

conn = sqlite3.connect('portfolio.db')
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("Tables in DB:", tables)

cursor.execute("SELECT * FROM portfolio")
print("Portfolio data:", cursor.fetchall())

cursor.execute("SELECT * FROM portfolio_assets")
print("Portfolio Assets data:", cursor.fetchall())

conn.close()
