# Congressional Trading ML

This project uses machine learning to predict whether a congressional stock trade will beat the S&P 500. Members of Congress must publicly report stock trades within 45 days under the STOCK Act (2012). We collect those disclosures, attach historical stock prices, engineer features, and train classifiers and regressors to see if the trades are predictable.

## Research Questions

1. Can ML models predict whether a congressional trade will outperform the S&P 500?
2. Do congressional trades generally beat the market?
3. Which features best predict a winning trade? Committee power, sector alignment, trade timing, or market momentum?

## Results

**Congressional trades do not reliably beat the market.** Mean excess return over 90 days is –0.2%. Only 43.5% of trades outperform the index.

### Classification: did the trade outperform by >5% in 90 days?

| Model | ROC-AUC (80/20) | CV Mean AUC |
|---|---|---|
| Logistic Regression | 0.612 | 0.588 ± 0.054 |
| Random Forest | 0.643 | 0.615 ± 0.057 |
| XGBoost | 0.642 | 0.606 ± 0.052 |

### Regression: predict exact excess return over 90 days

| Model | R² | Directional Accuracy |
|---|---|---|
| Ridge | 0.034 | 50.7% |
| Random Forest | 0.091 | 51.4% |
| XGBoost | 0.130 | 54.7% |

Stock and sector momentum dominate feature importance. Broad market context explains more variance than any congressional-specific feature.

## Dataset

| Source | What it contains |
|---|---|
| [Capitol Trades](https://www.capitoltrades.com/trades) | Congressional trade disclosures (9,675 trades, 2023–2025) |
| Yahoo Finance | Daily stock prices for 1,112 tickers |
| Yahoo Finance | 11 SPDR sector ETF price histories |
| [congress-legislators](https://github.com/unitedstates/congress-legislators) | Member metadata (538 members, 3,908 committee memberships) |

## Features

| Feature | What it means |
|---|---|
| `committee_power_score` | How powerful the member's committees are (max 5 for Finance, Appropriations, Intelligence) |
| `committee_sector_alignment` | 1 if the member's committee oversees the stock's sector |
| `sector` | GICS sector of the traded stock |
| `trade_size_bucket` | Size category: small / medium_small / medium / large / very_large / mega |
| `spy_momentum_30d/90d` | S&P 500 return in the 30 or 90 days before the trade |
| `stock_momentum_30d/60d` | The stock's own return in the 30 or 60 days before the trade |
| `sector_etf_momentum_30d` | The sector ETF's return in the 30 days before the trade |

## Setup

**Requirements:** Python 3.9+, Google Chrome, ChromeDriver matching your Chrome version.

Download ChromeDriver from [chromedriver.chromium.org](https://chromedriver.chromium.org/) and put it on your PATH.

```bash
git clone https://github.com/wdcannella/congressional-trading-ml.git
cd congressional-trading-ml
pip install -r requirements.txt
```

## Usage

### Step 1 — Scrape trades

```bash
python src/capitoltrades_scraper.py --max-pages 363
```

This opens a Chrome browser, visits Capitol Trades, and saves all trade disclosures to `data/capitoltrades_data.csv`. Use `--headless` to run without opening a window. This may take upwards of 2 hours

### Step 2 — Download stock prices

```bash
python src/download_stock_prices.py
```

Downloads historical daily prices for every ticker in the trades data, the S&P 500 benchmark (SPY), and 11 sector ETFs from Yahoo Finance. Saves to `data/stock_prices.pkl`, `data/sp500_prices.csv`, and `data/sector_etf_prices.pkl`.

### Step 3 — Process and merge data

```bash
python src/process_data.py
```

Cleans the raw scrape, matches politicians to official member records, and builds all features. Saves to `data/trades_with_features.csv`.

### Step 4 — Calculate returns

```bash
python src/calculate_returns.py
```

For each trade, looks up the stock price at the trade date and 30, 60, and 90 days later. Computes excess return vs. the S&P 500 and creates the binary target variable. Saves to `data/trades_with_returns.csv`.

### Step 5 — Train models

```bash
# classification: predict whether trade beats market by >5%
python src/models.py --model logistic_regression
python src/models.py --model random_forest
python src/models.py --model xgboost

# regression: predict exact 90-day excess return
python src/models.py --model xgboost --task regression
```

### Notebooks

Open the notebooks in order for a guided walkthrough:

| Notebook | What it does |
|---|---|
| `notebooks/data_processing.ipynb` | Shows the cleaning and feature engineering steps |
| `notebooks/exploratory_analysis.ipynb` | Charts and summary stats on the trade data |
| `notebooks/ml_models.ipynb` | Model training, evaluation, and feature importance |

## Project Structure

```
congressional-trades-ml/
├── data/                        # raw and processed data files
├── notebooks/                   # exploratory notebooks
├── src/
│   ├── capitoltrades_scraper.py # scrapes capitoltrades.com
│   ├── download_stock_prices.py # downloads prices from Yahoo Finance
│   ├── process_data.py          # cleans and merges all data
│   ├── calculate_returns.py     # computes returns and targets
│   └── models.py                # trains and evaluates ML models
├── requirements.txt
└── README.md
```

## References

1. Ziobrowski et al. (2011) — "Abnormal Returns from the Common Stock Investments of the U.S. Senate"
2. Eggers & Hainmueller (2013) — "Capitol Losses: The Mediocre Performance of Congressional Stock Portfolios"
3. Tahoun (2014) — "The Role of Stock Ownership by US Members of Congress on the Market for Political Favors"
4. Zhou & Wei (2024) — "Political Power and Profitable Trades in the US Congress"
5. Mintarya et al. (2023) — "Machine Learning Approaches in Stock Market Prediction"

## Author

William Cannella
Advanced Machine Learning, Spring 2026
Professor: Ming Jin