# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 14:25:12 2025

@author: piyush
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import os.path
from YahooOHLCFetcher import *

class MomentumPredictors:
    """
    Computes momentum predictors for cross-sectional strategies as described in:
    "Building Cross-Sectional Systematic Strategies By Learning to Rank"
    
    Implements three types of predictors:
    1. Raw cumulative returns
    2. Normalized returns
    3. MACD-based indicators
    """
    
    def __init__(self, prices: pd.DataFrame, daily_returns: pd.DataFrame):
        """
        Initialize with price and return data.
        
        Args:
            prices: DataFrame with dates as index and tickers as columns
            daily_returns: DataFrame of daily returns with same structure as prices
        """
        self.prices = prices
        self.daily_returns = daily_returns
        self.volatility = self._compute_volatility()
        
    def _compute_volatility(self) -> pd.DataFrame:
        """Compute 63-day exponentially weighted volatility"""
        return self.daily_returns.ewm(span=63).std()
    
    def _get_returns(self, lookback_days: int) -> pd.DataFrame:
        """Compute returns over specified lookback period"""
        return self.prices.pct_change(lookback_days)
    
    def compute_raw_returns(self) -> Dict[str, pd.DataFrame]:
        """
        Compute raw cumulative returns over 3, 6, and 12-month periods
        
        Returns:
            Dictionary with keys 'ret_3m', 'ret_6m', 'ret_12m'
        """
        trading_days_per_month = 21
        return {
            'ret_3m': self._get_returns(3 * trading_days_per_month),
            'ret_6m': self._get_returns(6 * trading_days_per_month),
            'ret_12m': self._get_returns(12 * trading_days_per_month)
        }
    
    def compute_normalized_returns(self) -> Dict[str, pd.DataFrame]:
        """
        Compute volatility-normalized returns over 3, 6, and 12-month periods
        
        Returns:
            Dictionary with keys 'norm_ret_3m', 'norm_ret_6m', 'norm_ret_12m'
        """
        raw_rets = self.compute_raw_returns()
        normalized = {}
        
        for period, ret_df in raw_rets.items():
            # Scale daily volatility to match return period
            scale_factor = np.sqrt(int(period.split('_')[1][:-1]))  # Extract number from key
            normalized[f'norm_{period}'] = ret_df / (self.volatility * scale_factor)
            
        return normalized
    
    def _ewma(self, span: int) -> pd.DataFrame:
        """Compute exponentially weighted moving average"""
        return self.prices.ewm(span=span).mean()
    
    def compute_macd_indicators(self) -> Dict[str, pd.DataFrame]:
        """
        Compute MACD-based indicators as described in Baz et al. (2015)
        
        Returns:
            Dictionary containing:
            - Raw intermediate signals for each time scale combination
            - Final composite signal
        """
        # Define time scales (short and long) as in the paper
        short_scales = [8, 16, 32]
        long_scales = [24, 48, 96]
        
        macd_signals = {}
        
        # Compute intermediate signals for each time scale combination
        for s, l in zip(short_scales, long_scales):
            # Compute MACD (difference between short and long EMAs)
            macd = self._ewma(s) - self._ewma(l)
            
            # Volatility normalize (63-day rolling std of prices)
            price_vol = self.prices.rolling(63).std()
            macd_norm = macd / price_vol
            
            # Standardize by 252-day rolling std of MACD values
            macd_std = macd_norm.rolling(252).std()
            final_signal = macd_norm / macd_std
            
            macd_signals[f'macd_{s}_{l}'] = final_signal
            
            # Also store intermediate signals for past periods
            for lookback in [21, 63, 126, 252]:  # 1, 3, 6, 12 months
                macd_signals[f'macd_{s}_{l}_lag{lookback}'] = final_signal.shift(lookback)
        
        # Compute composite signal (sum of response function outputs)
        # Here we use identity function for phi() as in the paper
        composite_signal = sum(macd_signals[f'macd_{s}_{l}'] 
                              for s, l in zip(short_scales, long_scales))
        macd_signals['macd_composite'] = composite_signal
        
        return macd_signals
    
    def compute_all_predictors(self) -> pd.DataFrame:
        """
        Compute all predictors and combine into a single DataFrame with multi-index
        (date, ticker) and each predictor as a column.
        
        Returns:
            DataFrame with all predictors
        """
        # Compute each type of predictor
        raw_rets = self.compute_raw_returns()
        norm_rets = self.compute_normalized_returns()
        macd_indicators = self.compute_macd_indicators()
        
        # Combine all predictors
        all_predictors = {**raw_rets, **norm_rets, **macd_indicators}
        
        # Stack each DataFrame to long format and combine
        predictor_dfs = []
        for name, df in all_predictors.items():
            stacked = df.stack().to_frame(name)
            predictor_dfs.append(stacked)
        
        # Combine all predictors
        full_df = pd.concat(predictor_dfs, axis=1)
        
        return full_df

# Example usage
if __name__ == "__main__":
    # Load your price data (example)
    # prices = pd.read_csv('your_price_data.csv', index_col=0, parse_dates=True)
    # daily_rets = prices.pct_change()
    
    # For demonstration, create mock data
    dates = pd.date_range('1995-01-01', '2023-12-31')
    # tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    ohlc_path = r'Data\sp500_combined_ohlc_data.csv'
    if os.path.exists(ohlc_path ):
        ohlc_data = pd.read_csv(ohlc_path)
        prices = ohlc_data.loc[:,['Date','ticker','close']]
        prices = prices.pivot(index='Date', columns='ticker', values='close')
        
    else:
        START_DATE = "1995-01-01"
        END_DATE = (datetime.today() -  relativedelta(months=1, day=31)).strftime('%Y-%m-%d')
        sp500_hist = pd.read_excel(r'Data\SP500_MonthEnd_Constituents_'+datetime(1998, 1, 1).date().strftime('%Y%m%d')+'.xlsx', index_col=0)
        all_tickers = list(pd.unique(sp500_hist .values.ravel()))
        TICKERS = [x for x in all_tickers if not (isinstance(x, str) and x == np.nan)]

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
        
        combined_df = pd.concat(data.values())
        
        # Reorder columns to have ticker first
        cols = ['ticker'] + [col for col in combined_df.columns if col != 'ticker']
        prices = combined_df[cols]
        
    rets = prices.pct_change()
        
    # Initialize and compute predictors
    predictor = MomentumPredictors(prices, rets)
    all_predictors = predictor.compute_all_predictors()
    
    all_predictors = all_predictors.reset_index()
    all_predictors ['Date'] = pd.to_datetime(all_predictors ['Date'])
    all_predictors = all_predictors .set_index(['Date','ticker'])
    
    
    all_predictors.to_pickle(r'Data\LTR_Momentum_Indicators.pkl')
    print("Available predictors:")
    print(all_predictors.columns.tolist())
    
    # Example: Get predictors for a specific date
    example_date = '2023-01-31'
    print(f"\nPredictors for {example_date}:")
    print(all_predictors.loc[example_date].head())
