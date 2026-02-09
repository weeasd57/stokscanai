import os
from dotenv import load_dotenv
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.historical import CryptoHistoricalDataClient

load_dotenv()

def test_alpaca_crypto():
    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_SECRET_KEY")
    
    if not key or not secret:
        print("Keys missing")
        return

    client = CryptoHistoricalDataClient(api_key=key, secret_key=secret)
    
    symbols = ["BTC/USD", "ETH/USD"]
    print(f"Testing symbols: {symbols}")
    
    request_params = CryptoBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Hour,
        limit=10
    )
    
    try:
        bars = client.get_crypto_bars(request_params)
        df = bars.df
        print("Bars found:")
        print(df.head())
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_alpaca_crypto()
