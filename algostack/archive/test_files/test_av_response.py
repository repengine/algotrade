#!/usr/bin/env python3
"""Test Alpha Vantage API responses to understand the data format."""

import os
import requests
import json

def test_av_responses():
    """Test Alpha Vantage API responses."""
    
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY', '991AR2LC298IGMX7')
    
    # Test daily data
    print("Testing TIME_SERIES_DAILY...")
    daily_url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=SPY&apikey={api_key}&datatype=json"
    
    response = requests.get(daily_url)
    data = response.json()
    
    print(f"Response keys: {list(data.keys())}")
    
    if 'Time Series (Daily)' in data:
        time_series = data['Time Series (Daily)']
        dates = list(time_series.keys())[:2]
        print(f"\nFirst date: {dates[0]}")
        print(f"Data fields: {list(time_series[dates[0]].keys())}")
        print(f"Sample data: {time_series[dates[0]]}")
    
    # Test adjusted daily data
    print("\n" + "="*60)
    print("Testing TIME_SERIES_DAILY_ADJUSTED...")
    adjusted_url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=SPY&apikey={api_key}&datatype=json"
    
    response2 = requests.get(adjusted_url)
    data2 = response2.json()
    
    print(f"Response keys: {list(data2.keys())}")
    
    if 'Time Series (Daily)' in data2:
        time_series2 = data2['Time Series (Daily)']
        dates2 = list(time_series2.keys())[:2]
        print(f"\nFirst date: {dates2[0]}")
        print(f"Data fields: {list(time_series2[dates2[0]].keys())}")
        print(f"Sample data: {time_series2[dates2[0]]}")
    
    # Test intraday
    print("\n" + "="*60)
    print("Testing TIME_SERIES_INTRADAY...")
    intraday_url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=SPY&interval=5min&apikey={api_key}&datatype=json&outputsize=compact"
    
    response3 = requests.get(intraday_url)
    data3 = response3.json()
    
    print(f"Response keys: {list(data3.keys())}")
    
    for key in data3.keys():
        if 'Time Series' in key:
            print(f"\nTime series key: '{key}'")
            time_series3 = data3[key]
            timestamps = list(time_series3.keys())[:2]
            print(f"First timestamp: {timestamps[0]}")
            print(f"Data fields: {list(time_series3[timestamps[0]].keys())}")
            print(f"Sample data: {time_series3[timestamps[0]]}")
            break

if __name__ == "__main__":
    test_av_responses()