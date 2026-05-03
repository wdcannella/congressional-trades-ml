"""
Data Processing Pipeline

Handles state extraction issues and column merging problems.

Usage:
    python src/process_data.py
"""

import pandas as pd
import numpy as np
from datetime import datetime
import re
import os


#maps top-level committee IDs to the GICS sectors they oversee
COMMITTEE_SECTOR_MAP = {
    'SSAF': ['Consumer Staples'],
    'HSAG': ['Consumer Staples'],
    'SSAP': ['Industrials'],
    'HSAP': ['Industrials'],
    'SSAS': ['Industrials'],
    'HSAS': ['Industrials'],
    'SSBK': ['Financials'],
    'HSBA': ['Financials'],
    'SSBU': ['Financials'],
    'HSBU': ['Financials'],
    'SSCM': ['Communication Services', 'Technology', 'Consumer Discretionary'],
    'HSSY': ['Technology'],
    'SSHR': ['Healthcare', 'Consumer Discretionary'],
    'HSED': ['Healthcare', 'Consumer Discretionary'],
    'SSEG': ['Energy', 'Utilities', 'Materials'],
    'HSII': ['Energy', 'Materials'],
    'HSIF': ['Energy', 'Utilities', 'Communication Services', 'Technology', 'Healthcare'],
    'SSEV': ['Utilities', 'Industrials', 'Materials'],
    'HSPW': ['Industrials'],
    'SSFI': ['Financials', 'Healthcare', 'Consumer Staples'],
    'HSWM': ['Financials', 'Healthcare', 'Consumer Staples'],
    'SSFR': ['Industrials'],
    'HSFA': ['Industrials'],
    'HSHM': ['Industrials', 'Technology'],
    'SLIN': ['Technology', 'Industrials'],
    'HLIG': ['Technology', 'Industrials'],
    'SSJU': ['Financials'],
    'HSJU': ['Financials'],
    'SSSB': ['Financials'],
    'HSSM': ['Financials'],
    'SSVA': ['Healthcare', 'Industrials'],
    'HSVR': ['Healthcare', 'Industrials'],
    'SSGA': ['Healthcare'],
    'JCSE': ['Financials'],
    'JSEC': ['Financials'],
    'JSTX': ['Financials'],
}


def load_raw_data():
    print("Loading raw data...")
    trades = pd.read_csv('data/capitoltrades_data.csv')
    members = pd.read_csv('data/legislators-current.csv')
    committees = pd.read_csv('data/committee-membership-current.csv')

    print(f"Loaded {len(trades)}  trades")
    print(f"Loaded {len(members)}  members")
    print(f"Loaded {len(committees)}  committee memberships")
    return trades, members, committees


def fix_trade_states(trades):
    print("\nFixing state extraction and cleaning names...")

    if 'state' in trades.columns:
        print(f"Existing state values count: {len(trades['state'].unique())}")

    def clean_name_and_extract_state(text):
        if pd.isna(text):
            return '', ''

        name = str(text).strip()

        #remove artifacts like "OtherSenateME" that the scraper sometimes leaves in the name field
        artifacts = [r'OtherSenate[A-Z]{2}', r'OtherHouse[A-Z]{2}', r'Senate[A-Z]{2}', r'House[A-Z]{2}']
        for artifact in artifacts:
            name = re.sub(artifact, '', name, flags=re.IGNORECASE).strip()

        state = ''
        words = name.split()
        if words:
            last_word = words[-1].strip()
            if len(last_word) == 2 and last_word.isupper():
                if last_word not in ['US', 'NA', 'VP', 'JR', 'II', 'IV']:

                    state =  last_word
                    name = " ".join(words[:-1]).strip()

        name =re.sub(r'\s+(Jr|Sr|II|III|IV)\.?$', '', name, flags=re.IGNORECASE).strip()
        return name, state

    cleaned_data = trades['politician_name'].apply(clean_name_and_extract_state)
    trades['politician_name_clean'] = [x[0] for x in cleaned_data]
    extracted_states = [x[1] for x in cleaned_data]

    if 'state' not in trades.columns:
        trades['state'] = extracted_states
    else:
        trades['state'] =trades['state'].fillna('').replace('', np.nan)
        trades['state'] =trades['state'].fillna(pd.Series(extracted_states))
        trades['state'] =trades['state'].fillna('')

    print(f"Sample cleaned names: {trades['politician_name_clean'].head(5).tolist()}")
    print(f"Empty states: {(trades['state'] == '').sum()}")
    return trades


def process_committees(committees):
    """Process committee data to create useful features."""

    print("\nProcessing committee data...")

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

    committee_features['committee_power_score']  = [
        calc_power_score(b) for b in committee_features.index
    ]

    print(f"Created features for {len(committee_features)} members")
    return  committee_features


def process_members(members, committee_features):
    """Process member metadata."""
    print("\nProcessing member metadata...")

    members['birthday'] = pd.to_datetime(members['birthday'])
    members['age'] =datetime.now().year - members['birthday'].dt.year

    members = members.merge(
        committee_features,
        left_on='bioguide_id',
        right_index=True,
        how='left'
    )


    #fill missing
    members['num_committees'] = members['num_committees'].fillna(0)
    members['is_committee_leader'] = members['is_committee_leader'].fillna(0)
    members['committee_power_score'] = members['committee_power_score'].fillna(0)

    leadership_bioguides = [
        'S000148',
        'M000355',
        'T000250',
        'D000563',
    ]

    members['is_leadership'] = members['bioguide_id'].isin(leadership_bioguides).astype(int)

    member_cols = [
        'bioguide_id', 'first_name', 'last_name', 'full_name',
        'party', 'state', 'type', 'age', 'gender',
        'num_committees', 'is_committee_leader', 'committee_power_score',
        'is_leadership'
    ]

    return members[member_cols]


def create_name_variations(name):
    if pd.isna(name):
        return []
    parts = str(name).strip().split()
    variations = []
    if len(parts) >=  2:
        first = parts[0]
        last= parts[-1]
        variations.append(f"{first} {last}")
        variations.append(f"{last}, {first}")
        variations.append(last)
    return variations


def merge_trades_with_members(trades, members):
    """
    Merge trades with member metadata.
    """
    print("\nMerging trades with member data...")

    name_col = 'politician_name_clean' if 'politician_name_clean' in trades.columns else 'politician_name'

    trades['last_name'] = trades[name_col].str.split().str[-1].str.upper().str.strip()
    trades['first_name'] = trades[name_col].str.split().str[0].str.upper().str.strip()

    members['last_name_upper'] = members['last_name'].str.upper().str.strip()
    members['first_name_upper'] = members['first_name'].str.upper().str.strip()
    members['full_name_upper'] = members['full_name'].str.upper().str.strip()
    members['chamber_mapped']= members['type'].map({'sen': 'Senate', 'rep': 'House'})

    print("Strategy 1: last_name + first_name + chamber")
    merged = trades.merge(
        members,
        left_on=['last_name', 'first_name', 'chamber'],
        right_on=['last_name_upper', 'first_name_upper', 'chamber_mapped'],
        how='left',
        suffixes=('_trade', '_member')
    )

    unmatched_mask = merged['bioguide_id'].isna()
    match_rate = (1 - unmatched_mask.sum() / len(merged)) * 100
    print(f"Match rate: {match_rate:.1f}%")

    if unmatched_mask.any():
        print("Strategy 2: politician_name_clean + chamber")
        unmatched_indices = merged[unmatched_mask].index
        unmatched_trades = trades.loc[unmatched_indices].copy()
        unmatched_trades['name_upper'] = unmatched_trades[name_col].str.upper().str.strip()

        rematch = unmatched_trades.merge(
            members,
            left_on=['name_upper', 'chamber'],
            right_on=['full_name_upper', 'chamber_mapped'],
            how='left',
            suffixes=('_trade', '_member')
        )

        for col in members.columns:
            if col == 'chamber':
                continue
            if col not in rematch.columns:
                continue
            target_col = col + '_member' if col + '_member' in merged.columns else col
            if target_col in merged.columns:
                merged.loc[unmatched_indices, target_col] = rematch[col].values
            else:
                if col in ['bioguide_id', 'full_name', 'age', 'gender', 'num_committees',
                           'is_committee_leader', 'committee_power_score', 'is_leadership']:
                    merged.loc[unmatched_indices, col] = rematch[col].values

        unmatched_mask = merged['bioguide_id'].isna()
        match_rate = (1 - unmatched_mask.sum() / len(merged)) * 100
        print(f"Match rate after Strategy 2: {match_rate:.1f}%")

    #try last name + state + chamber
    if unmatched_mask.any():
        print("Strategy 3: last_name + state + chamber")
        unmatched_indices = merged[unmatched_mask].index
        unmatched_trades = trades.loc[unmatched_indices].copy()

        rematch = unmatched_trades.merge(
            members,
            left_on=['last_name', 'state', 'chamber'],
            right_on=['last_name_upper', 'state', 'chamber_mapped'],
            how='left',
            suffixes=('_trade', '_member')
        )

        for col in members.columns:
            if col == 'chamber':
                continue
            if col not in rematch.columns:
                continue
            target_col = col + '_member' if col + '_member' in merged.columns else col
            if target_col in merged.columns:
                merged.loc[unmatched_indices, target_col] = merged.loc[unmatched_indices, target_col].fillna(
                    pd.Series(rematch[col].values, index=unmatched_indices)
                )
            else:
                if col in ['bioguide_id', 'full_name', 'age', 'gender', 'num_committees',
                           'is_committee_leader', 'committee_power_score', 'is_leadership']:
                    merged.loc[unmatched_indices, col] = rematch[col].values

        unmatched_mask = merged['bioguide_id'].isna()
        match_rate = (1 - unmatched_mask.sum() / len(merged)) * 100
        print(f"Match rate after Strategy 3: {match_rate:.1f}%")

    if unmatched_mask.any():
        print("\nStill unmatched trades:")
        cols_to_show = ['politician_name', name_col, 'state_trade', 'chamber', 'last_name', 'first_name']
        cols_to_show = [c for c in cols_to_show if c in merged.columns]
        print(merged[unmatched_mask][cols_to_show].drop_duplicates())

    if 'party_member' in merged.columns and 'party_trade' in merged.columns:
        merged['party'] = merged['party_member'].fillna(merged['party_trade'])
    elif 'party_member' in merged.columns:
        merged['party'] = merged['party_member']
    elif 'party_trade' in merged.columns:
        merged['party'] = merged['party_trade']

    if 'state_member' in merged.columns:
        merged['state'] = merged['state_member'].fillna(merged['state_trade'])

    return merged


def create_trade_features(df):
    print("\nCreating trade features...")

    df['published_date'] = pd.to_datetime(df['published_date'], format='%d %b %Y', errors='coerce')
    df['traded_date'] = pd.to_datetime(df['traded_date'], format='%d %b %Y', errors='coerce')

    df['trade_year'] = df['traded_date'].dt.year
    df['trade_month'] = df['traded_date'].dt.month
    df['trade_quarter'] = df['traded_date'].dt.quarter

    def parse_size(size_str):
        if pd.isna(size_str) or size_str == 'N/A':
            return np.nan
        try:
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
            return np.mean(numbers)
        except:
            return np.nan

    df['trade_amount'] = df['size'].apply(parse_size)

    def size_bucket(amount):
        if pd.isna(amount):
            return 'unknown'
        if amount <= 15000:
            return 'small'
        elif amount <= 50000:
            return 'medium_small'
        elif amount <= 100000:
            return 'medium'
        elif amount <= 500000:
            return 'large'
        elif amount <= 5000000:
            return 'very_large'
        else:
            return 'mega'

    df['trade_size_bucket'] = df['trade_amount'].apply(size_bucket)
    df['is_buy'] = (df['transaction_type'] == 'buy').astype(int)
    df['is_sell'] = (df['transaction_type'] == 'sell').astype(int)
    df['is_joint_ownership'] = (df['owner'].str.contains('Joint', case=False, na=False)).astype(int)

    return df


def load_sector_data():
    path = 'data/ticker_sectors.csv'
    if not os.path.exists(path):
        print("\nWARNING: data/ticker_sectors.csv not found.")
        print("Run download_stock_prices.py first to generate sector data.")
        print("Sector and alignment features will be set to 'Unknown' / 0.")
        return {}
    sector_df = pd.read_csv(path)
    sector_map = dict(zip(sector_df['ticker'], sector_df['sector']))
    print(f"\nLoaded sector data for {len(sector_map)} tickers")
    return sector_map


def create_sector_features(df, sector_map, committees):
    print("\nCreating sector and committee-alignment features...")

    df['sector'] = df['ticker'].map(sector_map).fillna('Unknown')
    known = (df['sector'] != 'Unknown').sum()
    print(f"Sector assigned: {known}/{len(df)} trades ({known/len(df)*100:.1f}%)")

    #strip subcommittee suffix (e.g. 'SSFI01' -> 'SSFI') before looking up COMMITTEE_SECTOR_MAP
    committees = committees.copy()
    committees['top_committee'] = committees['committee_id'].str[:4]
    committees['covered_sectors'] = committees['top_committee'].map(
        lambda cid: COMMITTEE_SECTOR_MAP.get(cid, [])
    )

    member_sectors = (
        committees[committees['covered_sectors'].map(len) > 0]
        .explode('covered_sectors')
        [['bioguide', 'covered_sectors']]
        .drop_duplicates()
    )
    member_sector_set = (
        member_sectors
        .groupby('bioguide')['covered_sectors']
        .apply(set)
        .to_dict()
    )

    def alignment(row):
        bioguide = row.get('bioguide_id')
        sector = row.get('sector')
        if pd.isna(bioguide) or sector == 'Unknown':
            return 0
        return int(sector in member_sector_set.get(bioguide, set()))

    df['committee_sector_alignment'] = df.apply(alignment, axis=1)

    aligned = df['committee_sector_alignment'].sum()
    print(f"Committee-sector aligned trades: {aligned}/{len(df)} ({aligned/len(df)*100:.1f}%)")

    return df


def save_processed_data(df):
    print("\nSaving processed data...")

    desired_cols = [
        'politician_name', 'ticker', 'issuer_name',
        'traded_date', 'published_date', 'trade_year', 'trade_month', 'trade_quarter',
        'transaction_type', 'is_buy', 'is_sell',
        'trade_amount', 'size', 'price',
        'filed_after_days', 'owner', 'is_joint_ownership',
        'full_name', 'bioguide_id', 'party', 'state', 'chamber',
        'age', 'gender',
        'num_committees', 'is_committee_leader', 'committee_power_score',
        'is_leadership',
        'sector', 'committee_sector_alignment',
        'trade_size_bucket',
        'scraped_at'
    ]

    available_cols = [col for col in desired_cols if col in df.columns]
    df_final = df[available_cols].copy()

    print(f"\nBefore filtering: {len(df_final)} rows")
    df_final = df_final.dropna(subset=['ticker', 'traded_date'])
    print(f"After filtering: {len(df_final)} rows")

    output_path = 'data/trades_with_features.csv'
    df_final.to_csv(output_path, index=False)

    print(f"\nSaved to {output_path}")
    print(f"Columns: {len(df_final.columns)}")

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
    print("="*60)
    print("CONGRESSIONAL TRADING DATA PROCESSING PIPELINE")
    print("="*60)
    
    # load
    trades, members, committees = load_raw_data()
    
    #fix states
    trades = fix_trade_states(trades)
    
    #process
    committee_features = process_committees(committees)
    members_processed = process_members(members, committee_features)
    
    # Merge
    merged = merge_trades_with_members(trades, members_processed)
    
    # features
    merged = create_trade_features(merged)

    #sector + committee features
    sector_map = load_sector_data()
    merged = create_sector_features(merged, sector_map, committees)

    #save
    final_df = save_processed_data(merged)

    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"Output: data/trades_with_features.csv")

    return final_df


if __name__ == "__main__":
    df = main()