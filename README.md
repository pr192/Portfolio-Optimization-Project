
# 💼 Portfolio Risk & Return Prediction System

## 📊 Project Overview
This project is developed as part of a milestone-based client assignment focusing on **portfolio management, risk assessment, and return prediction** using real cryptocurrency data (Bitcoin & Ethereum).  
It automates financial analysis, detects portfolio risks, and provides predictive insights — helping investors make data-driven decisions.

---

## 🚀 Objectives
1. Analyze the historical performance of assets (BTC & ETH).
2. Evaluate portfolio risk using defined risk rules.
3. Predict future returns using machine learning.
4. Store and visualize insights through automated workflows.
5. Perform stress testing and apply risk-parity adjustments.

---

## 🧩 Milestone Summary

### 🧱 **Milestone 1: Data Acquisition & Preparation**
- Collected real BTC and ETH price data (CSV format).
- Cleaned, preprocessed, and validated datasets.
- Calculated **daily returns** and **portfolio-level metrics**.

**Output:**
- Cleaned data files ready for analysis (`BTC-USD.csv`, `ETH-USD.csv`).

---

### ⚙️ **Milestone 2: Portfolio Risk Analysis**
- Implemented 3 key **risk identification rules**:
  - **Volatility Check** (High market fluctuation)
  - **Beta Sensitivity Rule** (Asset correlation to ETH)
  - **Asset Concentration Risk** (Diversification issue)
- Stored all risk metrics and statuses (`PASS`/`FAIL`) in an SQLite database.
- Configured **email alert system** for automatic notifications on risk failures.

**Technologies:** `Python`, `SQLite`, `smtplib`, `pandas`, `numpy`

---

### 🔮 **Milestone 3: Return Prediction & Stress Testing**
#### Part 1: Predictive Modeling
- Used **Linear Regression** to forecast:
  - Individual asset returns (BTC, ETH)
  - Portfolio-level returns
- Visualized real vs. predicted returns using **matplotlib** charts.

#### Part 2: Stress Testing & Risk-Parity Rule
- Implemented stress testing to simulate adverse market scenarios.
- Applied **risk-parity strategy** to rebalance portfolio weights.
- Stored all outputs and metrics in the database.

**Technologies:** `scikit-learn`, `matplotlib`, `numpy`, `pandas`

---

### 🧠 **Milestone 4: Final Integration & Visualization**
- Combined all scripts into a complete workflow.
- Built a simple **Streamlit UI** for live project demonstration:
  - Upload datasets
  - Run prediction models
  - View graphs, risk metrics, and alerts
- Prepared a professional presentation (`Project_Presentation.pptx`).

**Technologies:** `Streamlit`, `SQLite`, `Python`, `matplotlib`

---

## 🧮 Workflow Diagram
Data Collection → Risk Analysis → Return Prediction → Stress Testing → Visualization & Alerts

---

## 🧰 Tools & Technologies Used
| Category | Tools |
|-----------|-------|
| Programming | Python |
| Data Analysis | Pandas, NumPy |
| ML/Prediction | Scikit-learn |
| Database | SQLite |
| Visualization | Matplotlib, Streamlit |
| Communication | smtplib (Email Alerts) |

---

## 🗂️ Project Structure
Portfolio-Optimization-Project/
│
├── Milestone-1/
│   └── data_preprocessing.py
│
├── Milestone-2/
│   └── risk_checker.py
│
├── Milestone-3/
│   ├── predictor.py
│   └── stress_test.py
│
├── Milestone-4/
│   └── app.py
│
├── data/
│   ├── BTC-USD.csv
│   ├── ETH-USD.csv
│
├── risk.db
├── Project_Presentation.pptx
└── README.md

---

## 📈 Sample Output Charts
- Portfolio Return Prediction  
- Risk Metrics Summary  
- Stress Test Behavior (Before vs After Adjustment)

---


> _“A portfolio that manages its risk intelligently can survive any market storm.”_
