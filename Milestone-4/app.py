# ---------- app.py ----------
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import smtplib
from email.mime.text import MIMEText
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.set_page_config(page_title="Portfolio Risk Dashboard", layout="wide")

# ---------------- HEADER ----------------
st.title("📊 Portfolio Risk & Return Analyzer")
st.markdown("### A live demo integrating all milestones (Data → Risk → Prediction → Alert)")

# ---------------- FILE UPLOAD ----------------
st.sidebar.header("📂 Upload Your Data")
btc_file = st.sidebar.file_uploader("Upload BTC-USD.csv", type=["csv"])
eth_file = st.sidebar.file_uploader("Upload ETH-USD.csv", type=["csv"])

if btc_file and eth_file:
    btc = pd.read_csv(btc_file)
    eth = pd.read_csv(eth_file)

    # Handle ETH dataset format differences
    if 'Close' not in eth.columns and 'Price' in eth.columns:
        eth.rename(columns={'Price': 'Close'}, inplace=True)

    # Convert numeric columns
    btc['Close'] = pd.to_numeric(btc['Close'], errors='coerce')
    eth['Close'] = pd.to_numeric(eth['Close'], errors='coerce')

    # Calculate returns
    btc['Return'] = btc['Close'].pct_change()
    eth['Return'] = eth['Close'].pct_change()

    # ---------------- PORTFOLIO RETURN ----------------
    st.subheader("💰 Portfolio & Individual Returns")

    weights = [0.6, 0.4]
    portfolio_return = (weights[0] * btc['Return'] + weights[1] * eth['Return']).dropna()

    col1, col2 = st.columns(2)
    with col1:
        st.line_chart(btc['Return'], use_container_width=True)
        st.caption("BTC Daily Returns")

    with col2:
        st.line_chart(eth['Return'], use_container_width=True)
        st.caption("ETH Daily Returns")

    st.line_chart(portfolio_return, use_container_width=True)
    st.caption("Portfolio Combined Return")

    # ---------------- RISK CALCULATIONS ----------------
    st.subheader("⚠️ Risk Analysis")

    volatility_btc = btc['Return'].std()
    volatility_eth = eth['Return'].std()
    cov_matrix = np.cov(btc['Return'].dropna(), eth['Return'].dropna())
    beta_btc = cov_matrix[0, 1] / np.var(eth['Return'].dropna())
    concentration_risk = max(weights)

    st.write(f"**Volatility (BTC):** {volatility_btc:.4f}")
    st.write(f"**Volatility (ETH):** {volatility_eth:.4f}")
    st.write(f"**Beta (BTC):** {beta_btc:.4f}")
    st.write(f"**Asset Concentration Risk:** {concentration_risk:.2f}")

    # ---------------- DATABASE STORAGE ----------------
    conn = sqlite3.connect("risk_data.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS risk_metrics(
            metric TEXT,
            value REAL,
            status TEXT
        )
    """)
    conn.commit()

    def check_rule(metric, value, threshold, higher_is_bad=True):
        if higher_is_bad:
            return "FAIL" if value > threshold else "PASS"
        else:
            return "FAIL" if value < threshold else "PASS"

    rules = {
        "Volatility_BTC": (volatility_btc, 0.03, True),
        "Volatility_ETH": (volatility_eth, 0.03, True),
        "Beta_BTC": (beta_btc, 1.0, True),
        "Asset_Concentration": (concentration_risk, 0.7, True)
    }

    failed_rules = []
    for metric, (val, thresh, hib) in rules.items():
        status = check_rule(metric, val, thresh, hib)
        c.execute("INSERT INTO risk_metrics VALUES (?, ?, ?)", (metric, val, status))
        if status == "FAIL":
            failed_rules.append(metric)
    conn.commit()

    st.success("✅ Risk metrics saved to database (risk_data.db)")

    # ---------------- LINEAR REGRESSION ----------------
    st.subheader("🔮 Return Prediction (Linear Regression)")

    X = np.arange(len(portfolio_return)).reshape(-1, 1)
    y = portfolio_return.values
    model = LinearRegression().fit(X, y)
    next_day = model.predict([[len(portfolio_return) + 1]])

    st.metric("Predicted Next-Day Portfolio Return", f"{next_day[0]*100:.2f}%")

    fig, ax = plt.subplots()
    ax.plot(X, y, label="Actual", linewidth=2)
    ax.plot(X, model.predict(X), '--', label="Predicted Trend", color='orange')
    ax.legend()
    st.pyplot(fig)

    # ---------------- EMAIL ALERT ----------------
    st.subheader("📧 Email Risk Alert")

    sender_email = "pg0009605@gmail.com"
    sender_password = "qbwmbytlpnwbgkyg"
    receiver_email = "p20992987@gmail.com"

    if failed_rules:
        st.error("🚨 Risk Threshold Breach Detected!")
        msg = MIMEText("Risk Alert! The following metrics failed:\n" + "\n".join(failed_rules))
        msg['Subject'] = "Risk Checker Alert"
        msg['From'] = sender_email
        msg['To'] = receiver_email

        if st.button("Send Email Alert"):
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(msg)
            st.success(f"📨 Email alert sent to {receiver_email}")
    else:
        st.success("All metrics are within safe limits ✅")

    conn.close()

else:
    st.info("📤 Please upload both BTC-USD.csv and ETH-USD.csv to begin analysis.")
