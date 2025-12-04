import yfinance as yf
import sys

def test_ticker(symbol):
    with open("test_output.txt", "w") as f:
        f.write(f"Testing symbol: {symbol}\n")
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            if not info:
                f.write("Info is empty.\n")
            else:
                f.write(f"Info has {len(info)} keys.\n")
                f.write(f"Has 'symbol': {'symbol' in info}\n")
                f.write(f"Has 'regularMarketPrice': {'regularMarketPrice' in info}\n")
                f.write(f"Has 'marketCap': {'marketCap' in info}\n")
                f.write(f"Has 'longName': {'longName' in info}\n")
                f.write(f"Has 'currentPrice': {'currentPrice' in info}\n")
                f.write(f"First 10 keys: {list(info.keys())[:10]}\n")
            
            history = stock.history(period="1y")
            if history.empty:
                f.write("History is empty.\n")
            else:
                f.write(f"History fetched successfully. Shape: {history.shape}\n")
            
            # Check news
            news = stock.news
            f.write(f"News fetched. Count: {len(news) if news else 0}\n")
            
            # Check financials
            income = stock.income_stmt
            f.write(f"Income stmt fetched. Empty: {income.empty if income is not None else 'None'}\n")
            
            balance = stock.balance_sheet
            f.write(f"Balance sheet fetched. Empty: {balance.empty if balance is not None else 'None'}\n")
            
            cashflow = stock.cashflow
            f.write(f"Cashflow fetched. Empty: {cashflow.empty if cashflow is not None else 'None'}\n")
                
        except Exception as e:
            f.write(f"Error: {e}\n")

if __name__ == "__main__":
    test_ticker("TCS.NS")
