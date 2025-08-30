import os
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path

import pandas as pd
import requests
from openai import OpenAI

class CryptoCurrency(Enum):
    SOLANA = "SOL-USD"
    ETHEREUM = "ETH-USD"
    BITCOIN = "BTC-USD"

if not Path(".env").exists():
    API_KEY = os.getenv("OPEN_API_KEY")
else:
    from dotenv import load_dotenv
    load_dotenv()
    API_KEY = os.getenv("OPEN_API_KEY")

# ------ USER DEFINED VARIABLES ----------
PROMPT = (
    "Identify support and resistance levels for Solana based on the provided OHLCV data and technical indicators."
    " Write a short summary in this format: Solana is currently trading at $X. The daily support level is $Y and the daily resistance level is $Z."
)
gpt_version = "5"
symbol = CryptoCurrency.SOLANA.value
days_to_analyze = 30
granularity = 3600
limit = 200
# --------- END USER DEFINED VARIABLES ---------


# ---------- FETCH CANDLE DATA ----------
def get_coinbase_candles(symbol=symbol, granularity=granularity, limit=limit):
    """
    Fetch OHLCV candlestick data from Coinbase
    params:
        symbol (str): The cryptocurrency symbol to fetch data for.
        granularity (int): The granularity of the candlestick data (in seconds).
        limit (int): The number of data points to retrieve.
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(seconds=granularity * limit)

    url = f"https://api.exchange.coinbase.com/products/{symbol}/candles"
    params = {
        "granularity": granularity,  # e.g. 3600 = 1h candles
        "start": start.isoformat(),
        "end": end.isoformat(),
    }

    response = requests.get(url, params=params)
    data = response.json()

    df = pd.DataFrame(data, columns=["time", "low", "high", "open", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.sort_values("time")
    return df


# ---------- ADD INDICATORS ----------
def add_indicators(df):
    df["SMA20"] = df["close"].rolling(window=20).mean()
    df["EMA20"] = df["close"].ewm(span=20, adjust=False).mean()

    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    short_ema = df["close"].ewm(span=12, adjust=False).mean()
    long_ema = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = short_ema - long_ema
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    return df


# ---------- SEND TO CHATGPT ----------
def analyze_with_chatgpt(df, prompt=PROMPT):
    client = OpenAI(api_key=API_KEY)

    # Convert last 100 rows to JSON to keep size reasonable
    data_sample = df.tail(100).to_dict(orient="records")

    completion = client.chat.completions.create(
        model=f"gpt-{gpt_version}",
        messages=[
            {"role": "system", "content": f"{prompt}"},
            {"role": "user", "content": f"\n\nHere is the OHLCV + indicators data:\n{data_sample}"},
        ],
    )

    return completion.choices[0].message.content


# ---------- MAIN ----------
def get_resistance_support(symbol=symbol, granularity=granularity, limit=limit):
    candles = get_coinbase_candles(symbol=symbol, granularity=granularity, limit=limit)
    candles = add_indicators(candles)

    analysis = analyze_with_chatgpt(candles)
    message = f"{analysis}"
    return message