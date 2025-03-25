# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 02:22:45 2025

@author: piyus
"""
import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import time
from datetime import datetime, timedelta,date
from dateutil.relativedelta import relativedelta
import os

def get_adjusted_ohlc(
    tickers: List[str],
    start_date: str,
    end_date: str,
    interval: str = '1d',
    batch_size: int = 100,
    max_retries: int = 3,
    pause_between_batches: float = 1.0
) -> Dict[str, pd.DataFrame]:
    """
    Fetch adjusted OHLC prices from Yahoo Finance for multiple tickers with batch processing.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        interval: Data interval ('1d', '1wk', '1mo')
        batch_size: Number of tickers per batch
        max_retries: Maximum number of retries for failed requests
        pause_between_batches: Seconds to pause between batches (avoid rate limiting)
        
    Returns:
        Dictionary with tickers as keys and DataFrames of adjusted OHLC data as values
    """
    # Validate dates
    try:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        if start_dt >= end_dt:
            raise ValueError("start_date must be before end_date")
    except ValueError as e:
        raise ValueError(f"Invalid date format: {e}. Use 'YYYY-MM-DD' format.")
    
    # Initialize data storage and tracking
    ohlc_data = {}
    failed_tickers = []
    processed_tickers = []
    total_batches = (len(tickers) // batch_size + (1 if len(tickers) % batch_size != 0 else 0))
    
    print(f"\nStarting download for {len(tickers)} tickers in {total_batches} batches")
    print(f"Date range: {start_date} to {end_date}")
    print("="*60)
    
    # Process in batches
    for batch_num, i in enumerate(range(0, len(tickers), batch_size), 1):
        batch = tickers[i:i + batch_size]
        remaining_tickers = len(tickers) - i - len(batch)
        
        print(f"\nBatch {batch_num}/{total_batches}: Processing {len(batch)} tickers "
              f"({remaining_tickers} remaining)")
        print("-"*50)
        
        for attempt in range(max_retries):
            try:
                # Print attempt info
                if attempt > 0:
                    print(f"Retry attempt {attempt + 1} for batch {batch_num}")
                
                # Download batch data
                print(f"Downloading data for: {', '.join(batch)}")
                data = yf.download(
                    tickers=batch,
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    group_by='ticker',
                    progress=False,
                    threads=True
                )
                
                # Process each ticker in the batch
                for ticker in batch:
                    try:
                        if len(batch) == 1:
                            # Single ticker case
                            df = data.copy()
                        else:
                            # Multi-ticker case
                            if ticker not in data.columns.get_level_values(0):
                                raise ValueError(f"No data found for {ticker}")
                            df = data[ticker].copy()
                        
                        if df.empty:
                            raise ValueError(f"Empty DataFrame for {ticker}")
                        
                        # Clean and adjust column names
                        df.columns = df.columns.str.lower()
                        
                        # Handle adjusted close
                        if 'adj close' not in df.columns:
                            df['adj close'] = df['close']
                        
                        # Calculate adjusted OHLC
                        adj_factor = df['adj close'] / df['close']
                        df['adj open'] = df['open'] * adj_factor
                        df['adj high'] = df['high'] * adj_factor
                        df['adj low'] = df['low'] * adj_factor
                        df['adj volume'] = df['volume'] / adj_factor
                        
                        # Keep only adjusted columns
                        adj_cols = ['adj open', 'adj high', 'adj low', 'adj close', 'adj volume']
                        df = df[adj_cols].rename(columns=lambda x: x.replace('adj ', ''))
                        
                        # Add ticker column
                        df['ticker'] = ticker
                        
                        # Store in dictionary
                        ohlc_data[ticker] = df
                        processed_tickers.append(ticker)
                        print(f"✓ Successfully processed {ticker}")
                        
                    except Exception as e:
                        print(f"✗ Failed to process {ticker}: {str(e)}")
                        if ticker not in failed_tickers:
                            failed_tickers.append(ticker)
                        if ticker in processed_tickers:
                            processed_tickers.remove(ticker)
                
                break  # Success - exit retry loop
            
            except Exception as e:
                print(f"Batch {batch_num} download failed: {str(e)}")
                if attempt == max_retries - 1:
                    print(f"Max retries reached for batch {batch_num}")
                    failed_tickers.extend([t for t in batch if t not in processed_tickers])
                time.sleep(2)  # Wait before retry
        
        # Pause between batches to avoid rate limiting
        if i + batch_size < len(tickers):
            time.sleep(pause_between_batches)
    
    # Print summary
    print("\n" + "="*60)
    print("Download Summary:")
    print(f"Successfully processed: {len(processed_tickers)}/{len(tickers)} tickers")
    print(f"Failed tickers: {failed_tickers if failed_tickers else 'None'}")
    
    return ohlc_data

def save_to_single_csv(data: Dict[str, pd.DataFrame], filename: str = 'yahoo_ohlc_data.csv'):
    """
    Save all OHLC data to a single CSV file with ticker identifiers.
    
    Args:
        data: Dictionary of DataFrames from get_adjusted_ohlc()
        filename: Output CSV filename
    """
    if not data:
        print("No data to save!")
        return
    
    # Combine all DataFrames
    combined_df = pd.concat(data.values())
    
    # Reorder columns to have ticker first
    cols = ['ticker'] + [col for col in combined_df.columns if col != 'ticker']
    combined_df = combined_df[cols]
    
    # Save to CSV
    combined_df.to_csv(filename, index=True)
    print(f"\nSaved combined data for {len(data)} tickers to {filename}")
    print(f"Total records: {len(combined_df):,}")

# Example usage
if __name__ == "__main__":
    # Configuration
   
    START_DATE = "1995-01-01"
    END_DATE = (datetime.today() -  relativedelta(months=1, day=31)).strftime('%Y-%m-%d')
    sp500_hist = pd.read_excel(r'G:\My Drive\Quant Code PK\Learning to Rank Model\SP500_MonthEnd_Constituents_'+datetime(1998, 1, 1).date().strftime('%Y%m%d')+'.xlsx', index_col=0)
    all_tickers = list(pd.unique(sp500_hist .values.ravel()))
    TICKERS = [x for x in all_tickers if not (isinstance(x, str) and x == np.nan)]


    OUTPUT_FILE = 'sp500_combined_ohlc_data.csv'
    
    # Fetch data
    print(f"Fetching adjusted OHLC data for {len(TICKERS)} tickers")
    start_time = time.time()
    
    ohlc_data = get_adjusted_ohlc(
        tickers=TICKERS,
        start_date=START_DATE,
        end_date=END_DATE,
        batch_size=50,  # Conservative batch size
        pause_between_batches=1.5
    )
    
    # Save to single CSV
    save_to_single_csv(ohlc_data, OUTPUT_FILE)
    
    elapsed = time.time() - start_time
    print(f"\nTotal execution time: {elapsed:.2f} seconds")
    
    # Display sample data
    # if ohlc_data:
    #     sample_ticker = list(ohlc_data.keys())[0]
    #     print(f"\nSample data for {sample_ticker}:")
    #     print(ohlc_data[sample_ticker].head())