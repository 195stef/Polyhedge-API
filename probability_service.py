from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import sqlite3
import requests
import time
import asyncio
import numpy as np
from datetime import datetime, timedelta
import os

app = FastAPI(title="Polyhedger BTC Forecaster API v2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Relative to workspace root if run from there
DB_PATH = "bitcoin_data.db"

# ─────────────────────────────────────────────────────────
# DATA SYNC
# ─────────────────────────────────────────────────────────

def sync_data():
    """Download missing 1-minute candles from Binance and store in SQLite."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''CREATE TABLE IF NOT EXISTS prices 
                   ("Open time" TEXT, Open REAL, High REAL, Low REAL, Close REAL, Volume REAL, symbol TEXT)''')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_time ON prices("Open time")')

    try:
        cursor = conn.cursor()
        cursor.execute('SELECT MAX("Open time") FROM prices')
        last_time = cursor.fetchone()[0]
        if last_time:
            last_ts = int(datetime.fromisoformat(last_time).timestamp() * 1000)
        else:
            # START FROM 2019 to ensure 6y history (today is 2026)
            print("Empty DB. Starting sync from 2019-01-01...")
            last_ts = int(datetime(2019, 1, 1).timestamp() * 1000)
    except Exception as e:
        print(f"Init Error: {e}. Starting from 2019-01-01...")
        last_ts = int(datetime(2019, 1, 1).timestamp() * 1000)

    current_ts = int(time.time() * 1000)

    while last_ts < current_ts - 60000:
        url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&startTime={last_ts + 60000}&limit=1000"
        try:
            res = requests.get(url, timeout=10).json()
            if not res or (isinstance(res, dict) and 'code' in res):
                print(f"Binance error or no more data: {res}")
                break
            batch = [{
                'Open time': datetime.fromtimestamp(k[0] / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                'Open': float(k[1]), 'High': float(k[2]), 'Low': float(k[3]),
                'Close': float(k[4]), 'Volume': float(k[5]), 'symbol': 'BTCUSDT'
            } for k in res]
            pd.DataFrame(batch).to_sql('prices', conn, if_exists='append', index=False)
            last_ts = res[-1][0]
            print(f"Sync: up to {batch[-1]['Open time']}")
            # Small sleep to be nice to API
            time.sleep(0.1)
        except Exception as e:
            print(f"Sync error: {e}")
            break

    conn.close()
    print(f"✅ Sync complete at {datetime.now().strftime('%H:%M:%S')}")


# ─────────────────────────────────────────────────────────
# CORE FORECASTING ENGINE
# ─────────────────────────────────────────────────────────

def load_closes(days_of_history: int) -> pd.Series:
    """Load closing prices for the last N days (5-minute granularity for precision)."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cutoff = (datetime.now() - timedelta(days=days_of_history)).strftime('%Y-%m-%d %H:%M:%S')
        df = pd.read_sql(
            f'SELECT "Open time", Close FROM prices WHERE "Open time" >= ? ORDER BY "Open time" ASC',
            conn, params=(cutoff,)
        )
        conn.close()
        if df.empty:
            return pd.Series(dtype=float)
        
        df['Open time'] = pd.to_datetime(df['Open time'])
        df = df.set_index('Open time')
        # Resample to 5-minute for high precision
        df_5m = df['Close'].resample('5min').last().dropna()
        return df_5m
    except Exception as e:
        print(f"Load Closes Error: {e}")
        return pd.Series(dtype=float)


def compute_forecast(closes: pd.Series, days_ahead: float) -> dict | None:
    """
    Vectorized computation of forward returns.
    """
    intervals_ahead = int(days_ahead * 24 * 12) # 12 intervals per hour (5 min)
    if len(closes) < intervals_ahead + 10:
        return None

    vals = closes.values
    if len(vals) <= intervals_ahead:
        return None
        
    start_prices = vals[:-intervals_ahead]
    end_prices = vals[intervals_ahead:]
    returns = (end_prices / start_prices) - 1
    
    s = pd.Series(returns)
    return {
        "p10": float(s.quantile(0.10)),
        "p25": float(s.quantile(0.25)),
        "p50": float(s.quantile(0.50)),
        "p75": float(s.quantile(0.75)),
        "p90": float(s.quantile(0.90)),
        "mean": float(s.mean()),
        "samples": int(len(s))
    }


def prob_above(returns: np.ndarray | pd.Series, required_return: float) -> float:
    """What fraction of historical windows achieved >= required_return?"""
    if len(returns) == 0:
        return 0.0
    hits = (returns >= required_return).sum()
    return round(float(hits / len(returns)) * 100, 1)


# ── LIVE PRICE ──
def get_live_price():
    """Get the absolute latest price from Binance ticker."""
    try:
        res = requests.get("https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT", timeout=5).json()
        return float(res['price'])
    except:
        return None

@app.get("/price")
async def price():
    p = get_live_price()
    return {"price": p, "timestamp": datetime.now().isoformat()}


# ─────────────────────────────────────────────────────────
# API ENDPOINTS
# ─────────────────────────────────────────────────────────

@app.get("/predict")
async def predict(expiry: float, strike: float = None):
    if expiry <= 0 or expiry > 365:
        return {"error": "Please use an expiry value between 0 and 365 days"}

    # Try loading 6 years, fallback to 2 years, then fallback to anything we have
    closes = load_closes(days_of_history=2190)
    if closes.empty:
        closes = load_closes(days_of_history=730)
    if closes.empty:
        closes = load_closes(days_of_history=30) # Minimum 30 days to give a result
        
    if closes.empty:
        return {"error": "Database is still syncing initial data. Please try again in 5 minutes."}

    # IMPORTANT: Use live price as the anchor
    live_p = get_live_price()
    current_price = live_p if live_p else float(closes.iloc[-1])
    
    if strike is None:
        strike = current_price

    intervals_ahead = int(expiry * 24 * 12)
    vals = closes.values
    if len(vals) <= intervals_ahead or intervals_ahead == 0:
         return {"error": f"Not enough historical data synced yet for a {expiry}-day forecast. Currently have {len(vals)} data points."}
    
    # Vectorized calculation: (Price_at_T+N / Price_at_T) - 1
    start_prices = vals[:-intervals_ahead]
    end_prices = vals[intervals_ahead:]
    fwd_returns = (end_prices / start_prices) - 1
    
    # Required return to hit strike
    required_ret = (strike / current_price) - 1
    
    # Probability: percentage of historical periods that hit the required return
    probability = prob_above(fwd_returns, required_ret)
    
    # Calculate regime based on standard deviation of recent returns
    # We take the last 30 days of hourly returns
    recent_returns = (vals[-720:] / vals[-721:-1]) - 1 if len(vals) > 720 else fwd_returns
    volatility = np.std(recent_returns) * np.sqrt(24 * 365) * 100 # Annualized
    
    if volatility < 40:
        regime = "Low Vol"
    elif volatility < 65:
        regime = "Med-Low Vol"
    elif volatility < 85:
        regime = "Moderate Vol"
    elif volatility < 110:
        regime = "High Vol"
    else:
        regime = "Extreme Vol"
        
    req_move_str = f"{'+' if required_ret > 0 else ''}{required_ret * 100:.2f}%"

    return {
        "probability": probability,
        "regime": regime,
        "current_price": round(current_price, 2),
        "required_move": req_move_str,
        "samples": len(fwd_returns)
    }


@app.get("/status")
async def status():
    try:
        conn = sqlite3.connect(DB_PATH)
        res = pd.read_sql('SELECT COUNT(*) as count, MAX("Open time") as last FROM prices', conn)
        conn.close()
        live = get_live_price()
        return {
            "total_records": int(res.iloc[0, 0]),
            "last_db_record": res.iloc[0, 1],
            "live_binance_price": live,
            "status": "online"
        }
    except Exception as e:
        return {"status": "no_data", "error": str(e), "total_records": 0}


async def auto_sync_loop():
    while True:
        try:
            # Run the heavy sync in a separate thread to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, sync_data)
        except Exception as e:
            print(f"Auto-sync error: {e}")
        await asyncio.sleep(60)  # Every 1 minute to catch up faster


@app.on_event("startup")
async def startup_event():
    # Start the loop as a background task
    asyncio.create_task(auto_sync_loop())


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
