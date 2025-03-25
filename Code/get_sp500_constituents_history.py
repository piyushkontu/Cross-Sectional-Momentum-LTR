# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 13:48:40 2025

@author: piyus
"""
import pandas as pd
import requests
from datetime import datetime, timedelta,date
from dateutil.relativedelta import relativedelta
from collections import Counter

# Function to get the latest S&P 500 constituents from Wikipedia
def get_sp500_constituents():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url)
    tables = pd.read_html(response.text)
    sp500_table = tables[0]  # The first table contains the S&P 500 constituents
    return sp500_table['Symbol'].tolist()

# Function to get S&P 500 changes from Wikipedia
def get_sp500_changes():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url)
    tables = pd.read_html(response.text)

    # The "Changes" table is usually the second table on the page
    changes_table = tables[1]

    # Flatten multi-level column names
    changes_table.columns = [' '.join(col).strip() for col in changes_table.columns.values]

    # Rename columns for consistency
    changes_table.rename(columns={
        'Date Date': 'Date',
        'Added Ticker': 'AddTicker',
        'Added Security': 'AddName',
        'Removed Ticker': 'RemovedTicker',
        'Removed Security': 'RemovedName',
        'Reason Reason': 'Reason'
    }, inplace=True)

    # Clean the Date column
    changes_table['Date'] = pd.to_datetime(changes_table['Date'], errors='coerce')  # Coerce errors to handle invalid dates
    changes_table.dropna(subset=['Date'], inplace=True)  # Drop rows with invalid dates

    return changes_table

# Function to create monthly snapshots of S&P 500 constituents
def create_monthly_snapshots(start_date):
    sp500_changes = get_sp500_changes()
    sp500_changes = sp500_changes[sp500_changes['Date'] >= start_date]

    # Initialize the list of constituents at the start date
    current_constituents = get_sp500_constituents()
    snapshots = {}

    end_date = datetime.today() -  relativedelta(months=1, day=31)
    
    snapshots[end_date.strftime('%Y-%m-%d')] = current_constituents.copy()
    
    
    # Iterate through each month from the start date to the current date
    current_date = end_date
    while current_date >= start_date:
                
        month_end = current_date -  relativedelta(months=1, day=31)

        # Filter changes for the current month
        changes_in_month = sp500_changes[(sp500_changes['Date'] <= current_date) & (sp500_changes['Date'] >= month_end)]

        # Create a copy of the current constituents to avoid modifying the original list
        monthly_constituents = current_constituents.copy()

        
        # Apply changes for the current month
        for _, change in changes_in_month.iterrows():
            if pd.notna(change['AddTicker']) and change['AddTicker'] in monthly_constituents:
                monthly_constituents.remove(change['AddTicker'])
            if pd.notna(change['RemovedTicker']) and change['RemovedTicker'] not in monthly_constituents:
                monthly_constituents.append(change['RemovedTicker'])

        
                
        # Save the snapshot for the month-end
        snapshots[month_end.strftime('%Y-%m-%d')] = monthly_constituents.copy()
        
        # Update the current constituents for the next month
        current_constituents = monthly_constituents.copy()

        # Move to the previous month
        current_date = month_end 

    return snapshots

# Example usage
start_date = datetime(1998, 1, 1)
monthly_snapshots = create_monthly_snapshots(start_date)

# Convert the snapshots to a DataFrame for better visualization
snapshots_df = pd.DataFrame.from_dict(monthly_snapshots, orient='index')
snapshots_df.index.name = 'Date'
snapshots_df.to_excel('SP500_MonthEnd_Constituents_'+start_date.date().strftime('%Y%m%d')+'.xlsx')
# snapshots_df = snapshots_df.transpose()

# print(snapshots_df)