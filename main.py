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
def main(symbol=symbol, granularity=granularity, limit=limit):
    candles = get_coinbase_candles(symbol=symbol, granularity=granularity, limit=limit)
    candles = add_indicators(candles)

    analysis = analyze_with_chatgpt(candles)
    lines = "-------------------------------------------------------------------------"
    message = f"{analysis}\n{lines}\n End analysis at {datetime.now(timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M:%S')}"
    return message


analysis = main(symbol=symbol, granularity=granularity, limit=limit)

content = f"""
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <title>Imran A</title>
    <link rel="stylesheet" href="css/styles.css" />
    <link rel="icon" href="images/myicon.ico?v=2" />
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link
      href="https://fonts.googleapis.com/css2?family=Source+Code+Pro&display=swap"
      rel="stylesheet"
    />
  </head>

  <body>
    <div>
      <table id="topsection">
        <tr>
          <td>
            <img
              class="profilepic"
              src="images/profilepic.png?v=2"
              alt="profile pic Imran Aurangzeb"
              srcset=""
            />
          </td>
          <td>
            <h1 class="nametitle">Imran A</h1>
          </td>
        </tr>
      </table>
    </div>

    <hr size="3" />

    <div>
      <p>
        I began programming in BASIC on the Commodore 64 at the age of twelve.
        In my fifth decade, following a long hiatus during which I pursued a
        career in medicine, I chose to rekindle this passion. The journey has
        proved immensely rewarding.
      </p>
      <br />
    </div>

    <div class="two-columns">
      <div class="column">
        <h2>Column One</h2>
        <p>
          {analysis}
        </p>
      </div>
      <div class="column">
        <h2>Column Two</h2>
        <p>
          Suspendisse potenti. Nulla facilisi. Donec vitae eros vel sapien
          suscipit porttitor. Phasellus at augue a mi pretium bibendum.
          Suspendisse potenti. Nulla facilisi. Donec vitae eros vel sapien
          suscipit porttitor. Phasellus at augue a mi pretium bibendum.
        </p>
      </div>
    </div>
  </body>
</html>

"""


with open("index.html", "w") as f:
    f.write(content)