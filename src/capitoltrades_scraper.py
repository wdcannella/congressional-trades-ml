"""
Capitol Trades Scraper - v6 (FIXED FULL NAMES)
===============================================

Properly extracts full politician names and company names.
Converts relative dates like "Yesterday" to actual dates.
All data types are correct (state is string, not float).

Usage:
    python src/capitoltrades_scraper_v6.py --max-pages 10
"""

import argparse
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from datetime import datetime, timedelta
import re


class CapitolTradesScraper:
    """Scraper for Capitol Trades congressional trading data."""
    
    def __init__(self, headless=False):
        """Initialize the scraper."""
        self.url = "https://www.capitoltrades.com/trades"
        self.driver = None
        self.headless = headless
        self.trades = []
        self.scrape_date = datetime.now()
        
    def setup_driver(self):
        """Set up Selenium Chrome driver."""
        options = webdriver.ChromeOptions()
        if self.headless:
            options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        
        self.driver = webdriver.Chrome(options=options)
        self.driver.maximize_window()
        
    def clean_text(self, text):
        """Clean text by removing newlines and extra whitespace."""
        if not text:
            return ''
        cleaned = text.replace('\n', ' ').replace('\r', ' ')
        cleaned = ' '.join(cleaned.split())
        return cleaned.strip()
    
    def parse_relative_date(self, date_str):
        """
        Convert relative dates to actual dates.
        
        Args:
            date_str: String like "Yesterday", "Today", "10 Feb 2026", "09:01 Yesterday"
            
        Returns:
            str: Date in format "DD MMM YYYY" (e.g., "04 Mar 2026")
        """
        date_str = self.clean_text(date_str)
        
        if 'Yesterday' in date_str or 'yesterday' in date_str:
            yesterday = self.scrape_date - timedelta(days=1)
            return yesterday.strftime('%d %b %Y')
        
        if 'Today' in date_str or 'today' in date_str:
            return self.scrape_date.strftime('%d %b %Y')
        
        days_ago_match = re.search(r'(\d+)\s+days?\s+ago', date_str, re.IGNORECASE)
        if days_ago_match:
            days = int(days_ago_match.group(1))
            past_date = self.scrape_date - timedelta(days=days)
            return past_date.strftime('%d %b %Y')
        
        hours_ago_match = re.search(r'(\d+)\s+hours?\s+ago', date_str, re.IGNORECASE)
        if hours_ago_match:
            return self.scrape_date.strftime('%d %b %Y')
        
        date_pattern = re.search(r'(\d{1,2}\s+[A-Za-z]{3}\s+\d{4})', date_str)
        if date_pattern:
            return date_pattern.group(1)
        
        alt_pattern = re.search(r'([A-Za-z]{3}\s+\d{1,2}\s+\d{4})', date_str)
        if alt_pattern:
            try:
                parsed = datetime.strptime(alt_pattern.group(1), '%b %d %Y')
                return parsed.strftime('%d %b %Y')
            except:
                return alt_pattern.group(1)
        
        return date_str
        
    def wait_for_table_load(self, timeout=15):
        """Wait for the trades table to load."""
        try:
            wait = WebDriverWait(self.driver, timeout)
            wait.until(EC.presence_of_element_located((By.TAG_NAME, "table")))
            time.sleep(3)
            return True
        except TimeoutException:
            print("Timeout waiting for table to load")
            return False
            
    def scrape_current_page(self):
        """Scrape all trades from the current page."""
        page_trades = []
        
        try:
            table = self.driver.find_element(By.TAG_NAME, "table")
            rows = table.find_elements(By.TAG_NAME, "tr")[1:]
            
            print(f"Found {len(rows)} rows on this page")
            
            for idx, row in enumerate(rows):
                try:
                    cells = row.find_elements(By.TAG_NAME, "td")
                    
                    if len(cells) < 9:
                        continue
                    
                    politician_raw = self.clean_text(cells[0].text)
                    issuer_raw = self.clean_text(cells[1].text)
                    published_raw = self.clean_text(cells[2].text)
                    traded_raw = self.clean_text(cells[3].text)
                    
                    trade = {
                        'politician_name': self._extract_politician_name(politician_raw),
                        'party': self._extract_party(politician_raw),
                        'chamber': self._extract_chamber(politician_raw),
                        'state': self._extract_state(politician_raw),
                        'issuer_name': self._extract_issuer_name(issuer_raw),
                        'ticker': self._extract_ticker(issuer_raw),
                        'published_date': self.parse_relative_date(published_raw),
                        'traded_date': self.parse_relative_date(traded_raw),
                        'filed_after_days': self._extract_days(self.clean_text(cells[4].text)),
                        'owner': self.clean_text(cells[5].text),
                        'transaction_type': self.clean_text(cells[6].text).lower(),
                        'size': self.clean_text(cells[7].text),
                        'price': self.clean_text(cells[8].text),
                        'scraped_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    page_trades.append(trade)
                    
                except Exception as e:
                    print(f"Error scraping row {idx}: {str(e)}")
                    continue
            
        except Exception as e:
            print(f"Error scraping page: {str(e)}")
        
        return page_trades
    
    def _extract_ticker(self, text):
        """Extract ticker symbol from issuer cell."""
        if ':US' in text or ':' in text:
            match = re.search(r'([A-Z]+):', text)
            if match:
                return match.group(1)
        return ''
    
    def _extract_politician_name(self, text):
        """
        Extract full politician name from cell text.
        Format from site: "FirstName MiddleName LastName RepublicanSenateWV"
        (note: party/chamber/state are mashed together with no spaces)
        We want just the name part.
        """
        if not text:
            return ''
        
        # Keywords that mark end of name (these will be mashed with chamber/state)
        # Look for patterns like "Republican", "Democrat", etc.
        import re
        
        # Find where party name starts (first occurrence of Republican/Democrat/Independent)
        match = re.search(r'(Republican|Democrat|Independent)', text)
        
        if match:
            # Everything before the party is the name
            name = text[:match.start()].strip()
            return name
        
        # Fallback: return whole text if no party found
        return text.strip()
    
    
    def _extract_issuer_name(self, text):
        """
        Extract company/issuer name from cell text.
        Usually format: "Company Name Inc TICKER:US"
        We want everything before the ticker.
        """
        if not text:
            return ''
        
        # If there's a ticker, take everything before it
        if ':' in text:
            parts = text.split()
            name_parts = []
            for part in parts:
                if ':' in part:
                    break
                name_parts.append(part)
            return ' '.join(name_parts).strip()
        
        # Otherwise return first few words (company name)
        words = text.split()
        return ' '.join(words[:4]).strip() if len(words) > 0 else text
    
    
    def _extract_party(self, text):
        """Extract political party from politician cell."""
        if 'Republican' in text:
            return 'Republican'
        elif 'Democrat' in text:
            return 'Democrat'
        return ''
    
    def _extract_chamber(self, text):
        """
        Extract chamber (House/Senate) from politician cell.
        Format: "Name RepublicanSenateWV" (need to find Senate/House in middle)
        """
        if 'Senate' in text:
            return 'Senate'
        elif 'House' in text:
            return 'House'
        return ''
    
    def _extract_state(self, text):
        """
        Extract state abbreviation from politician cell.
        Format: "Name RepublicanSenateWV" (state is last 2 chars)
        """
        if not text:
            return ''
        
        # State is the LAST 2 characters of the text
        state = text[-2:].upper()
        
        # Verify it's actually a state code (2 uppercase letters)
        if len(state) == 2 and state.isalpha() and state.isupper():
            # Filter out obvious non-states
            if state not in ['AN', 'IC', 'AT']:  # End of "Republican", "Democratic", "Erat"
                return state
        
        return ''
    
    def _extract_days(self, text):
        """Extract number of days from 'Filed After' column."""
        match = re.search(r'(\d+)', text)
        if match:
            return int(match.group(1))
        return 0  # CHANGED from None to 0 to ensure int type
    
    def scroll_to_bottom(self):
        """Scroll to bottom of page to trigger any lazy loading."""
        self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)
    
    def go_to_page(self, page_num):
        """Navigate to a specific page number."""
        try:
            if page_num == 1:
                url = self.url
            else:
                url = f"{self.url}?page={page_num}"
            
            print(f"Loading: {url}")
            self.driver.get(url)
            
            if not self.wait_for_table_load():
                return False
            
            self.scroll_to_bottom()
            return True
            
        except Exception as e:
            print(f"Error navigating to page {page_num}: {str(e)}")
            return False
    
    def scrape_all(self, max_pages=10):
        """Scrape multiple pages of trades."""
        self.setup_driver()
        
        try:
            for page_num in range(1, max_pages + 1):
                print(f"\n{'='*60}")
                print(f"Scraping page {page_num}/{max_pages}")
                print(f"{'='*60}")
                
                if not self.go_to_page(page_num):
                    print(f"Failed to load page {page_num}")
                    break
                
                page_trades = self.scrape_current_page()
                
                if not page_trades:
                    print(f"No trades found on page {page_num}. Stopping.")
                    break
                
                self.trades.extend(page_trades)
                
                print(f"Collected {len(page_trades)} trades from page {page_num}")
                print(f"Total trades so far: {len(self.trades)}")
                
                time.sleep(3)
            
        except Exception as e:
            print(f"Error during scraping: {str(e)}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.driver.quit()
        
        df = pd.DataFrame(self.trades)
        
        # EXPLICITLY set column types to avoid pandas guessing wrong
        if not df.empty:
            df = df.astype({
                'politician_name': 'string',
                'party': 'string',
                'chamber': 'string',
                'state': 'string',  # FORCE string type
                'issuer_name': 'string',
                'ticker': 'string',
                'published_date': 'string',
                'traded_date': 'string',
                'filed_after_days': 'Int64',  # Nullable integer
                'owner': 'string',
                'transaction_type': 'string',
                'size': 'string',
                'price': 'string',
                'scraped_at': 'string'
            })
        
        return df
    
    def save_to_csv(self, df, filename='data/capitoltrades_data.csv'):
        """Save scraped data to CSV with proper quoting."""
        df.to_csv(filename, index=False, quoting=1)
        print(f"\nData saved to {filename}")
        print(f"Total records: {len(df)}")
        
        # Print data types for verification
        print("\nColumn Data Types:")
        print(df.dtypes)


def main():
    """Main function to run the scraper from command line."""
    parser = argparse.ArgumentParser(description='Scrape Capitol Trades data (v6 - Fixed Names)')
    parser.add_argument('--max-pages', type=int, default=5, 
                       help='Maximum number of pages to scrape (default: 5)')
    parser.add_argument('--headless', action='store_true',
                       help='Run browser in headless mode')
    parser.add_argument('--output', type=str, default='data/capitoltrades_data.csv',
                       help='Output CSV filename')
    
    args = parser.parse_args()
    
    print("Capitol Trades Scraper v6 (Fixed Full Names)")
    print("=" * 60)
    print(f"Scrape started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Max pages: {args.max_pages}")
    print(f"Headless mode: {args.headless}")
    print(f"Output file: {args.output}")
    print("=" * 60)
    
    scraper = CapitolTradesScraper(headless=args.headless)
    df = scraper.scrape_all(max_pages=args.max_pages)
    
    if not df.empty:
        scraper.save_to_csv(df, args.output)
        
        print("\n" + "=" * 60)
        print("SUMMARY STATISTICS")
        print("=" * 60)
        print(f"Total trades: {len(df)}")
        print(f"Unique politicians: {df['politician_name'].nunique()}")
        print(f"Unique tickers: {df['ticker'].nunique()}")
        print(f"\nDate range:")
        print(f"  Published: {df['published_date'].min()} to {df['published_date'].max()}")
        print(f"  Traded: {df['traded_date'].min()} to {df['traded_date'].max()}")
        print(f"\nTransaction types:")
        print(df['transaction_type'].value_counts())
        print(f"\nTop 10 most active traders:")
        print(df['politician_name'].value_counts().head(10))
        print(f"\nParty distribution:")
        print(df['party'].value_counts())
        print(f"\nChamber distribution:")
        print(df['chamber'].value_counts())
        print(f"\nState distribution (top 10):")
        print(df['state'].value_counts().head(10))
    else:
        print("\nNo data scraped. Check for errors above.")


if __name__ == "__main__":
    main()