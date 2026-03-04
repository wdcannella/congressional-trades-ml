# Congressional Trading ML Predictor

A machine learning project that analyzes congressional stock trades to predict stock performance and identify profitable trading patterns.

## Project Overview

This project develops ML models to predict stock returns based on congressional trading patterns disclosed under the STOCK Act (2012). By analyzing features such as congressional leadership positions, committee memberships, and trade timing, the model aims to identify which congressional trades are most likely to outperform the market.

### Research Questions
1. Can ML models predict stock returns based on congressional trading patterns?
2. Do congressional trades generally outperform the market (S&P 500)?
3. Which features (leadership position, committees, party control) best predict successful trades?

## Project Status
🚧 **In Development** - Currently in data collection phase

- [x] Project proposal completed
- [x] Web scraper developed
- [ ] Data collection (congressional trades 2020-2025)
- [ ] Stock price data collection
- [ ] Exploratory data analysis
- [ ] Feature engineering
- [ ] Baseline model development
- [ ] Advanced model development
- [ ] Model evaluation and backtesting

## Dataset

### Data Sources
- **Congressional Trades**: [Senate Electronic Financial Disclosure](https://efdsearch.senate.gov/)
- **Stock Prices**: Yahoo Finance API
- **Senator Metadata**: Congressional records (committees, party, tenure)

### Expected Data Size
- ~10,000-15,000 congressional transactions (2020-2025)
- Stock price data for ~500-1000 unique tickers
- Metadata for ~100 senators

## Methodology

### Machine Learning Pipeline
1. **Data Collection**: Web scraping congressional disclosures + stock price APIs
2. **Feature Engineering**: Leadership position, committee power, trade timing, market conditions
3. **Baseline Models**: Logistic Regression, Decision Trees
4. **Advanced Models**: Random Forest, XGBoost/LightGBM
5. **Evaluation**: Accuracy, Precision, Recall, F1, ROC-AUC, portfolio backtesting

### Target Variable
Binary classification: Does stock outperform S&P 500 by >5% at 90 days post-disclosure?

## Repository Structure

```
congressional-trading-ml/
├── README.md                 # Project overview (this file)
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
├── data/
│   ├── raw/                 # Original scraped data
│   ├── processed/           # Cleaned datasets
│   └── external/            # Stock prices, senator metadata
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_modeling.ipynb
├── src/
│   ├── __init__.py
│   ├── scraper.py           # Web scraping functions
│   ├── data_processing.py   # Data cleaning utilities
│   ├── features.py          # Feature engineering
│   └── models.py            # ML model implementations
├── results/
│   ├── figures/             # Visualizations
│   └── metrics/             # Model performance metrics
└── docs/
    └── proposal.pdf         # Project proposal
```

## Installation & Setup

### Prerequisites
- Python 3.8+
- Git
- Chrome browser (for Selenium web scraping)

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/congressional-trading-ml.git
cd congressional-trading-ml
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download ChromeDriver for Selenium:
   - Visit https://chromedriver.chromium.org/
   - Download version matching your Chrome browser
   - Place in project root or add to PATH

## Usage

### Data Collection
```bash
# Test congressional trade scraper
python src/capitoltrades_scraper.py --max-pages 3

# Run congressional trade scraper
python src/capitoltrades_scraper.py --max-pages 2899

# Run data processor
python src/process_data.py

# Download Stock Price Data
python src/download_stock_prices.py



### Exploratory Analysis

# Launch Jupyter notebook
jupyter notebook notebooks/data_collection.ipynb
```

### Model Training
```bash
# Train baseline models
#python src/models.py --model logistic_regression

# Train advanced models
#python src/models.py --model random_forest
```

## Key Findings

*To be updated as project progresses*

## References

1. Ziobrowski et al. (2011) - "Abnormal Returns from the Common Stock Investments of the U.S. Senate"
2. Eggers & Hainmueller (2013) - "Capitol Losses: The Mediocre Performance of Congressional Stock Portfolios"
3. Tahoun (2014) - "The Role of Stock Ownership by US Members of Congress on the Market for Political Favors"
4. Zhou & Wei (2024) - "Political Power and Profitable Trades in the US Congress"
5. Mintarya et al. (2023) - "Machine Learning Approaches in Stock Market Prediction"

## License

This project is for educational purposes as part of an Advanced Machine Learning course.

## Author

William Cannella  
Advanced Machine Learning - Spring 2026

## Acknowledgments

- Data sourced from public congressional financial disclosures
- Stock market data provided by Yahoo Finance API
- Project supervised by Ming Jin

---

**Disclaimer**: This project is for academic research only. Not financial advice. Congressional trading data is public information under the STOCK Act (2012).