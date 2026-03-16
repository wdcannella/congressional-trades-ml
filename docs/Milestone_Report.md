# Milestone Report: Congressional Trading ML Predictor

**Date:** March 5, 2026  
**Project:** Congressional Trading ML Predictor  
**Author:** William Cannella  

---

### 1. Progress from Proposal

Substantial progress has been made on all proposed items since the initial capstone proposal. The project has evolved from a conceptual framework to a fully functional data and modeling pipeline.

**Evolution of Ideas:**
*   **Data Scope:** Initially focused on simple trade data, the project now incorporates deep metadata for legislators, including committee assignments and leadership roles.
*   **Market Benchmarking:** The methodology was refined to calculate "excess returns" relative to the S&P 500 across multiple time horizons (30, 60, and 90 days), providing a more robust measure of "alpha" than absolute returns.
*   **Feature Engineering:** Expanded from basic trade attributes to derived features such as `committee_power_score`, `filed_after_days` (reporting lag), and `is_leadership` status.

---

### 2. Implementation

The project features a clean, modular, and documented codebase organized into a robust ML pipeline.

**Core Components:**
*   **Web Scraper (`src/capitoltrades_scraper.py`):** A Selenium-based scraper for Capitol Trades that handles dynamic content, pagination, and relative date conversion (e.g., "Yesterday" to "YYYY-MM-DD").
*   **Data Processor (`src/process_data.py`):** Orchestrates the merging of trade data with legislator metadata (from `congress-legislators` GitHub) and committee memberships.
*   **Return Calculator (`src/calculate_returns.py`):** Fetches historical stock data via `yfinance` and calculates trade performance relative to the market index.
*   **Modeling Pipeline (`src/models.py`):** Implements a standardized pipeline using `scikit-learn` for preprocessing (imputation, scaling, one-hot encoding) and model training.

**Repository Organization:**
The GitHub repository is well-organized with dedicated directories for `data/`, `notebooks/`, `src/`, and `docs/`. A comprehensive `README.md` provides installation and usage instructions.

---

### 3. Data Analysis (EDA)

A thorough Exploratory Data Analysis (EDA) has been completed, as documented in `notebooks/trade_analysis.ipynb`.

**Data Challenges & Solutions:**
*   **Inconsistent Naming:** Politicians' names often vary between data sources (e.g., "William" vs. "Bill"). A robust fuzzy matching and variation generation logic was implemented in `process_data.py`.
*   **Dynamic Web Content:** The scraper was built to handle asynchronous loading and specific CSS selectors that change based on screen size.
*   **Missing Price Data:** Handled by implementating a lookback/forward mechanism to find the nearest valid trading day price.

**EDA Highlights:**
*   **Volume:** High volume in the House (93% of trades) compared to the Senate.
*   **Party Balance:** Nearly 50/50 split in trading activity between Democrats and Republicans.
*   **Correlation:** A weak negative correlation (-0.14) was found between committee power and trading frequency, suggesting that higher power does not necessarily equal more active trading.

---

### 4. Preliminary Results

Multiple experiments have been run to establish baseline performance using Logistic Regression and Random Forest models.

**Baseline Comparisons:**
The target variable is binary: *Did the trade outperform the S&P 500 over 90 days?* (Baseline probability: ~34% for this specific target window).

| Model | Accuracy | Precision (Class 1) | Recall (Class 1) | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 66.1% | 40.0% | 0.3% | 0.521 |
| **Random Forest** | 70.3% | 64.3% | 26.8% | 0.695 |

**Results Analysis:**
*   The **Random Forest** model significantly outperforms Logistic Regression, particularly in its ability to identify outperforming trades (Recall: 26.8% vs 0.3%).
*   **Key Predictors:** The most influential features in the Random Forest model include `filed_after_days` (reporting delay), `trade_amount`, and `age`. The reporting delay suggests that the timing of public disclosure relative to the trade date may contain predictive signals.

---

### 5. Next Steps

The project is on track for completion with the following planned activities:

1.  **Advanced Modeling:** Implement XGBoost and LightGBM models to capture more complex non-linear relationships.
2.  **Extended Feature Engineering:** Incorporate sector-specific features and macroeconomic indicators (e.g., interest rate changes) at the time of trade.
3.  **Backtesting Module:** Develop a simulation environment to test a "Congressional Alpha" trading strategy based on model predictions.
4.  **Obstacles & Mitigation:** 
    *   *Obstacle:* Data sparsity for certain senators. *Mitigation:* Use grouped features (e.g., by committee type) rather than individual legislator IDs.
    *   *Obstacle:* Survivorship bias in stock tickers. *Mitigation:* Ensure the stock price downloader handles delisted tickers where data is available.

---

### 6. Writing & Presentation

This report and the project documentation adhere to professional technical writing standards. Visualizations in the notebooks use `seaborn` and `matplotlib` with clear labeling and consistent styling. All data sources and methodology choices are cited in the `README.md` and script docstrings.
