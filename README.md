# Congressional Trading ML Predictor

A machine learning project that analyzes congressional stock trades to predict stock performance and identify profitable trading patterns.

## Project Overview

This project develops ML models to predict stock returns based on congressional trading patterns disclosed under the STOCK Act (2012). By analyzing features such as congressional leadership positions, committee memberships, and trade timing, the model aims to identify which congressional trades are most likely to outperform the market.

### Research Questions
1. Can ML models predict stock returns based on congressional trading patterns?
2. Do congressional trades generally outperform the market (S&P 500)?
3. Which features (leadership position, committees, party control) best predict successful trades?

## Project Status
 **In Development** - Currently in data collection phase

**Completed**
- Project proposal
- Web scraper
- Data collection (congressional trades 2020-2025)
- Stock price data collection
- Baseline model development

**Planned**
- Feature engineering
- exploratory data analysis
- Advanced model development
- Model evaluation and backtesting

## Dataset

### Data Sources
- **Congressional Trades**: [Capitol Trades](https://www.capitoltrades.com/trades?pageSize=96)
- **Stock Prices**: Yahoo Finance API
- **Senator Metadata**: [congress-legislators github](https://github.com/unitedstates/congress-legislators?tab=readme-ov-file)

### Expected Data Size
- 7245 congressional transactions collected thus far
- ~20000 congressional transactions expected
- Stock price data for ~500-1000 unique tickers
- Metadata for 538 congresspeople

## Methodology

### Machine Learning Pipeline
1. **Data Collection**: Web scraping congressional disclosures + stock price APIs
2. **Feature Engineering**: Leadership position, committee power, trade timing, market conditions
3. **Baseline Models**: Logistic Regression, Random Forest
4. **Advanced Models**: TBD
5. **Evaluation**: TBD

### Target Variable
Binary classification: Does stock outperform S&P 500?

## Repository Structure

```
congressional-trading-ml/
├── README.md                 # Project overview (this file)
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore rules
├── data/                     # contains legislator metadata, raw trade data, stock prices, etc
├── notebooks/
│   ├── data_collection.ipynb
│   ├── trade_analysis.ipynb
│   └── modeling_analysis.ipynb
├── src/
│   ├── capitoltrades_scraper.py       # Web scraping 
│   ├── process_data.py                # Data processing
│   ├── download_stock_prices.py       # get stock prices
│   ├── calculate_returns.py           # calculate transaction returns
│   └── models.py                      # ML model implementations
└── docs/
    └── 5424 Capstone Proposal.pdf         # Project proposal
```

## Installation & Setup

### Installation Steps

1. Clone the repository:
```
git clone https://github.com/wdcannella/congressional-trading-ml.git
cd congressional-trading-ml
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Download ChromeDriver for Selenium:
   - Visit https://chromedriver.chromium.org/
   - Download version matching your Chrome browser
   - Place in project root or add to PATH

## Usage

### Data Collection
```
# Test congressional trade scraper
python src/capitoltrades_scraper.py --max-pages 3

# Run congressional trade scraper
python src/capitoltrades_scraper.py --max-pages 363

# Run data processor
python src/process_data.py

# Download Stock Price Data
python src/download_stock_prices.py
```


### Analysis
```
# Launch Jupyter notebooks
jupyter notebook notebooks/data_collection.ipynb
jupyter notebook notebooks/trade_analysis.ipynb
jupyter notebook notebooks/modeling_analysis.ipynb

```

### Model Training
```
# Train models
python src/models.py --model logistic_regression
python src/models.py --model random_forest

```

## Key Findings

- **Trading Balance**: Congressional trading is nearly equally divided between Republicans and Democrats.
- **House vs Senate**: The House of Representatives accounts for over 90% of all reported trades.
- **Market Performance**: Congressional trades show a modest average 30-day mean excess return of **+0.34%** relative to the S&P 500, but the median return is slightly negative (-0.09%).
- **Alpha Probability**: Approximately **49.2%** of congressional trades outperformed the market over a 30-day window.
- **Party Comparison**: In this dataset, Republican trades showed slightly higher mean and median excess returns than Democratic trades.
- **Committee Power**: There is no strong positive correlation between a member's committee power and their trading frequency.



## References

1. Ziobrowski et al. (2011) - "Abnormal Returns from the Common Stock Investments of the U.S. Senate"
2. Eggers & Hainmueller (2013) - "Capitol Losses: The Mediocre Performance of Congressional Stock Portfolios"
3. Tahoun (2014) - "The Role of Stock Ownership by US Members of Congress on the Market for Political Favors"
4. Zhou & Wei (2024) - "Political Power and Profitable Trades in the US Congress"
5. Mintarya et al. (2023) - "Machine Learning Approaches in Stock Market Prediction"


## Author

William Cannella  
Advanced Machine Learning - Spring 2026

## Acknowledgments

- Data sourced from public congressional financial disclosures
- Stock market data provided by Yahoo Finance API
- Project supervised by Ming Jin

---

**Disclaimer**: This project is for academic research only. Not financial advice. Congressional trading data is public information under the STOCK Act (2012).