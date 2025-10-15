
# My Portfolio Management Project - Milestone 2

## What I did in this milestone

So far, I created a solution to manage a cryptocurrency portfolio using deterministic data. Here's a simple summary of what I did:

---

### Data I Used
- Two assets: **BTC** and **ETH**
- Expected returns I assumed:
    - BTC = 0.12
    - ETH = 0.18
- Covariance matrix (for risk correlation):
    ```
    [[0.005, -0.010],
     [-0.010, 0.040]]
    ```

---

### Strategies I Implemented
1. **Equal Weight Rule**  
   I assigned 50% BTC and 50% ETH.

2. **Risk-Based Rule**  
   Lower risk assets get higher weights automatically.

3. **Performance-Based Rule**  
   Weights are proportional to the expected returns.

---

### Parallel Execution
I ran all strategies at the same time using Python's ThreadPoolExecutor to make it faster.

---

### Database Setup
I saved results in `portfolio.db` using SQLite, and the structure looks like this:

- **portfolio** table stores:  
    | portfolio_id | rule_name | total_return | total_risk |

- **portfolio_assets** table stores:  
    | asset_id | portfolio_id | currency | weight | expected_return | asset_risk |

---

### Comparison & Graph
I compared portfolio returns vs individual BTC and ETH returns and plotted a bar graph.  
Also, I exported this data into `portfolio_vs_assets.csv`.

Example table data:
| Name              | Return |
|-------------------|--------|
| equal_weight      | 0.1500 |
| risk_based        | 0.1360 |
| performance_based | 0.1560 |
| BTC               | 0.1200 |
| ETH               | 0.1800 |

---

### My Insights
- Performance-based strategy gives the highest return, but has more risk.
- Risk-based strategy keeps risk lower.
- Both portfolio strategies are better than holding BTC or ETH alone.
- The graph shows this visually and makes it easy to understand.

---

This makes my portfolio management solution reliable and easy to explain.  
I can extend it later by adding more strategies or assets.

---

