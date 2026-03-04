"""
Data Processing Pipeline - FIXED
=================================

Handles state extraction issues and column merging problems.

Usage:
    python src/process_data.py
"""

import pandas as pd
import numpy as np
from datetime import datetime
import re


def load_raw_data():
    """Load all raw data files."""
    print("Loading raw data...")
    
    trades = pd.read_csv('data/capitoltrades_data.csv')
    members = pd.read_csv('data/legislators-current.csv')
    committees = pd.read_csv('data/committee-membership-current.csv')
    
    print(f"Loaded {len(trades)} trades")
    print(f"Loaded {len(members)} members")
    print(f"Loaded {len(committees)} committee memberships")
    
    return trades, members, committees


def fix_trade_states(trades):
    """
    Fix state extraction issues from trade data.
    Re-extract from politician_name field.
    """
    print("\nFixing state extraction...")
    
    # First, let's see what we're working with
    print("Sample politician_name values:")
    print(trades['politician_name'].head(10).tolist())
    
    # Check if state is already in the data
    if 'state' in trades.columns:
        print(f"\nExisting state values: {trades['state'].value_counts().to_dict()}")
    
    def extract_state_fixed(text):
        """Extract 2-letter state code from end of text."""
        if pd.isna(text):
            return ''
        
        # Split and get last word
        words = str(text).split()
        if len(words) > 0:
            last_word = words[-1].strip()
            # Check if it's exactly 2 uppercase letters
            if len(last_word) == 2 and last_word.isupper():
                if last_word not in ['US', 'NA', 'VP']:
                    return last_word
        return ''
    
    # Only try to extract if state doesn't exist or is all empty
    if 'state' not in trades.columns or trades['state'].isna().all() or (trades['state'] == '').all():
        print("Attempting to extract state from politician_name...")
        trades['state_fixed'] = trades['politician_name'].apply(extract_state_fixed)
        trades['state'] = trades['state_fixed']
        trades = trades.drop(columns=['state_fixed'], errors='ignore')
    
    # Show what was extracted
    print(f"States after extraction: {trades['state'].value_counts().to_dict()}")
    print(f"Empty states: {(trades['state'] == '').sum()}")
    
    return trades


def process_committees(committees):
    """Process committee data to create useful features."""
    print("\nProcessing committee data...")
    
    # Define powerful committees
    powerful_committees = {
        'SSFI': 5,  # Senate Finance
        'SSAP': 5,  # Senate Appropriations  
        'SLIN': 5,  # Senate Intelligence
        'HSWM': 5,  # House Ways and Means
        'HSAP': 5,  # House Appropriations
        'HSIF': 4,  # House Energy and Commerce
        'SSBA': 4,  # Senate Banking
        'SSFR': 4,  # Senate Foreign Relations
    }
    
    # Aggregate by member
    committee_features = committees.groupby('bioguide').agg({
        'committee_id': 'count',
        'title': lambda x: 1 if any('Chairman' in str(t) or 'Ranking' in str(t) for t in x) else 0
    }).rename(columns={
        'committee_id': 'num_committees',
        'title': 'is_committee_leader'
    })
    
    # Calculate power score
    def calc_power_score(bioguide):
        member_committees = committees[committees['bioguide'] == bioguide]['committee_id'].values
        return sum(powerful_committees.get(c, 1) for c in member_committees)
    
    committee_features['committee_power_score'] = [
        calc_power_score(b) for b in committee_features.index
    ]
    
    print(f"Created features for {len(committee_features)} members")
    return committee_features


def process_members(members, committee_features):
    """Process member metadata."""
    print("\nProcessing member metadata...")
    
    # Calculate age
    members['birthday'] = pd.to_datetime(members['birthday'])
    current_year = datetime.now().year
    members['age'] = current_year - members['birthday'].dt.year
    
    # Merge committee features
    members = members.merge(
        committee_features,
        left_on='bioguide_id',
        right_index=True,
        how='left'
    )
    
    # Fill missing
    members['num_committees'] = members['num_committees'].fillna(0)
    members['is_committee_leader'] = members['is_committee_leader'].fillna(0)
    members['committee_power_score'] = members['committee_power_score'].fillna(0)
    
    # Leadership (manual - update this list)
    leadership_bioguides = [
        'S000148',  # Schumer
        'M000355',  # McConnell
        'T000250',  # Thune
        'D000563',  # Durbin
    ]
    members['is_leadership'] = members['bioguide_id'].isin(leadership_bioguides).astype(int)
    
    # Select columns
    member_cols = [
        'bioguide_id', 'first_name', 'last_name', 'full_name',
        'party', 'state', 'type', 'age', 'gender',
        'num_committees', 'is_committee_leader', 'committee_power_score',
        'is_leadership'
    ]
    
    return members[member_cols]


def create_name_variations(name):
    """Create different name formats for matching."""
    if pd.isna(name):
        return []
    
    # Handle formats like "Kelly Morrison" or "Markwayne Mullin"
    parts = str(name).strip().split()
    
    variations = []
    if len(parts) >= 2:
        first = parts[0]
        last = parts[-1]
        
        # Add variations
        variations.append(f"{first} {last}")  # Kelly Morrison
        variations.append(f"{last}, {first}")  # Morrison, Kelly
        variations.append(last)  # Morrison
        
    return variations


def merge_trades_with_members(trades, members):
    """
    Merge trades with member metadata.
    Uses multiple matching strategies.
    """
    print("\nMerging trades with member data...")
    
    # Prepare matching keys
    trades['last_name'] = trades['politician_name'].str.split().str[-1].str.upper().str.strip()
    trades['first_name'] = trades['politician_name'].str.split().str[0].str.upper().str.strip()
    
    members['last_name_upper'] = members['last_name'].str.upper().str.strip()
    members['first_name_upper'] = members['first_name'].str.upper().str.strip()
    members['chamber_mapped'] = members['type'].map({'sen': 'Senate', 'rep': 'House'})
    
    # Strategy 1: Match on last name + first name + chamber
    print("Trying match on: last_name + first_name + chamber")
    merged = trades.merge(
        members,
        left_on=['last_name', 'first_name', 'chamber'],
        right_on=['last_name_upper', 'first_name_upper', 'chamber_mapped'],
        how='left',
        suffixes=('_trade', '_member')
    )
    
    match_rate = (merged['bioguide_id'].notna().sum() / len(merged)) * 100
    print(f"Match rate (name + chamber): {match_rate:.1f}%")
    
    # If still poor, try with state too
    if match_rate < 50:
        print("Trying match on: last_name + state + chamber")
        merged = trades.merge(
            members,
            left_on=['last_name', 'state', 'chamber'],
            right_on=['last_name_upper', 'state', 'chamber_mapped'],
            how='left',
            suffixes=('_trade', '_member')
        )
        match_rate = (merged['bioguide_id'].notna().sum() / len(merged)) * 100
        print(f"Match rate (name + state + chamber): {match_rate:.1f}%")
    
    # Show unmatched
    if match_rate < 100:
        print("\nUnmatched trades:")
        # Use original column names (they may have _trade suffix after merge)
        cols_to_show = []
        for col in ['politician_name', 'state_trade', 'chamber', 'last_name', 'first_name']:
            if col in merged.columns:
                cols_to_show.append(col)
        
        if len(cols_to_show) > 0:
            unmatched = merged[merged['bioguide_id'].isna()][cols_to_show].drop_duplicates()
            print(unmatched)
        else:
            print(f"{(merged['bioguide_id'].isna()).sum()} unmatched trades")
    
    # Use trade party if member party is missing
    if 'party_member' in merged.columns and 'party_trade' in merged.columns:
        merged['party'] = merged['party_member'].fillna(merged['party_trade'])
    elif 'party_member' in merged.columns:
        merged['party'] = merged['party_member']
    elif 'party_trade' in merged.columns:
        merged['party'] = merged['party_trade']
    
    return merged


def create_trade_features(df):
    """Create features from trade data."""
    print("\nCreating trade features...")
    
    # Convert dates
    df['published_date'] = pd.to_datetime(df['published_date'], format='%d %b %Y', errors='coerce')
    df['traded_date'] = pd.to_datetime(df['traded_date'], format='%d %b %Y', errors='coerce')
    
    # Date features
    df['trade_year'] = df['traded_date'].dt.year
    df['trade_month'] = df['traded_date'].dt.month
    df['trade_quarter'] = df['traded_date'].dt.quarter
    
    # Convert size to numeric
    def parse_size(size_str):
        if pd.isna(size_str) or size_str == 'N/A':
            return np.nan
        try:
            # Handle formats like "15K–50K" or "1K-15K"
            size_str = str(size_str).replace('–', '-').replace(',', '')
            parts = size_str.split('-')
            
            numbers = []
            for part in parts:
                part = part.strip().upper()
                if 'K' in part:
                    numbers.append(float(part.replace('K', '')) * 1000)
                elif 'M' in part:
                    numbers.append(float(part.replace('M', '')) * 1000000)
                else:
                    numbers.append(float(part))
            
            return np.mean(numbers)  # Midpoint
        except:
            return np.nan
    
    df['trade_amount'] = df['size'].apply(parse_size)
    
    # Binary features
    df['is_buy'] = (df['transaction_type'] == 'buy').astype(int)
    df['is_sell'] = (df['transaction_type'] == 'sell').astype(int)
    df['is_joint_ownership'] = (df['owner'].str.contains('Joint', case=False, na=False)).astype(int)
    
    return df


def save_processed_data(df):
    """Save processed dataset."""
    print("\nSaving processed data...")
    
    # Define columns to keep (only those that exist)
    desired_cols = [
        # Trade info
        'politician_name', 'ticker', 'issuer_name',
        'traded_date', 'published_date', 'trade_year', 'trade_month', 'trade_quarter',
        'transaction_type', 'is_buy', 'is_sell',
        'trade_amount', 'size', 'price',
        'filed_after_days', 'owner', 'is_joint_ownership',
        
        # Member info
        'full_name', 'bioguide_id', 'party', 'state', 'chamber',
        'age', 'gender',
        
        # Committee/Leadership
        'num_committees', 'is_committee_leader', 'committee_power_score',
        'is_leadership',
        
        # Metadata
        'scraped_at'
    ]
    
    # Keep only columns that exist
    available_cols = [col for col in desired_cols if col in df.columns]
    df_final = df[available_cols].copy()
    
    # Remove rows with missing critical data
    print(f"\nBefore filtering: {len(df_final)} rows")
    
    # Only drop if ticker or traded_date is missing
    df_final = df_final.dropna(subset=['ticker', 'traded_date'])
    
    print(f"After filtering: {len(df_final)} rows")
    
    # Save
    output_path = 'data/trades_with_features.csv'
    df_final.to_csv(output_path, index=False)
    
    print(f"\nSaved to {output_path}")
    print(f"Columns: {len(df_final.columns)}")
    
    # Summary
    print("\n" + "="*60)
    print("PROCESSED DATA SUMMARY")
    print("="*60)
    print(f"Total trades: {len(df_final)}")
    print(f"Unique politicians: {df_final['politician_name'].nunique()}")
    print(f"Unique tickers: {df_final['ticker'].nunique()}")
    
    if 'party' in df_final.columns:
        print(f"\nParty distribution:")
        print(df_final['party'].value_counts())
    
    if 'chamber' in df_final.columns:
        print(f"\nChamber distribution:")
        print(df_final['chamber'].value_counts())
    
    print(f"\nTransaction type:")
    print(df_final['transaction_type'].value_counts())
    
    if 'bioguide_id' in df_final.columns:
        matched = df_final['bioguide_id'].notna().sum()
        print(f"\nMatched with member data: {matched}/{len(df_final)} ({matched/len(df_final)*100:.1f}%)")
    
    return df_final


def main():
    """Main processing pipeline."""
    print("="*60)
    print("CONGRESSIONAL TRADING DATA PROCESSING PIPELINE (FIXED)")
    print("="*60)
    
    # Load
    trades, members, committees = load_raw_data()
    
    # Fix states
    trades = fix_trade_states(trades)
    
    # Process
    committee_features = process_committees(committees)
    members_processed = process_members(members, committee_features)
    
    # Merge
    merged = merge_trades_with_members(trades, members_processed)
    
    # Features
    merged = create_trade_features(merged)
    
    # Save
    final_df = save_processed_data(merged)
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE!")
    print("="*60)
    print(f"Output: data/trades_with_features.csv")
    
    return final_df


if __name__ == "__main__":
    df = main()