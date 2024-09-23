from fastapi import FastAPI
import yfinance as yf
from datetime import datetime, timedelta

app = FastAPI()

# Function to find the last full trading day for a given ticker
def find_last_full_trading_day(ticker, days_to_check=7):
    # NSE trading hours (09:15 to 15:30 IST)
    start_time = "09:15"
    end_time = "15:29"
    
    # Check the last few days of data (to account for weekends, holidays)
    today = datetime.today()

    for i in range(days_to_check):
        # Get the date to check (go back one day at a time)
        check_date = today - timedelta(days=i)
        check_date_str = check_date.strftime(r'%Y-%m-%d')
        
        # Download minute-level data for that day (extend the end date by 1 day)
        # print(f"Checking date: {check_date_str}")
        data = yf.download(ticker, start=check_date_str, 
                           end=(check_date + timedelta(days=1)).strftime('%Y-%m-%d'), 
                           interval="1m")
        
        if len(data) == 0:
            # print(f"No data found for {check_date_str}. Likely a holiday.")
            continue  # No data for that day, likely a holiday or weekend

        # Check if the data covers the full trading day range
        first_time = data.index[0].strftime('%H:%M')
        last_time = data.index[-1].strftime('%H:%M')

        # print(f"First time in data: {first_time}, Last time in data: {last_time}")
        
        if first_time <= start_time and last_time >= end_time:
            # print(f"Last full trading day: {check_date_str}")
            return check_date_str  # Return the date once found

    return None  # Return None if no full trading day found in the range

# API endpoint to get the last full trading day for a given ticker
@app.get("/last_trading_day/")
def get_last_trading_day(ticker: str):
    last_trading_day = find_last_full_trading_day(ticker)
    if last_trading_day:
        return {"ticker": ticker, "last_full_trading_day": last_trading_day}
    else:
        return {"error": "Unable to find last full trading day for the ticker"}

