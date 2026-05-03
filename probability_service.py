import sqlite3
import pandas as pd
import requests
import time
from datetime import datetime
import os
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import threading

app = FastAPI()

# Configurazione CORS per permettere al frontend di chiamare l'API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = "bitcoin_data.db"

# Global sync diagnostics
sync_status = {
    "last_run": None,
    "last_result": "never_run",
    "last_error": None,
    "candles_added_last_run": 0,
    "total_runs": 0
}

def sync_data():
    """Funzione per scaricare i dati mancanti da Binance"""
    global sync_status
    sync_status["last_run"] = datetime.now().isoformat()
    sync_status["total_runs"] += 1

    conn = sqlite3.connect(DB_PATH)
    conn.execute('''CREATE TABLE IF NOT EXISTS prices 
                   ("Open time" TEXT, Open REAL, High REAL, Low REAL, Close REAL, Volume REAL, symbol TEXT)''')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_time ON prices("Open time")')
    
    try:
        last_time = pd.read_sql("SELECT MAX(\"Open time\") FROM prices", conn).iloc[0,0]
        if last_time:
            last_ts = int(datetime.fromisoformat(last_time).timestamp() * 1000)
            print(f"[SYNC] Ultimo dato in DB: {last_time}")
        else:
            print("[SYNC] Database vuoto. Parto dal 2024-01-01...")
            last_ts = int(datetime(2024, 1, 1).timestamp() * 1000)
    except Exception as e:
        print(f"[SYNC] Errore lettura DB: {e}")
        last_ts = int(datetime(2024, 1, 1).timestamp() * 1000)

    current_ts = int(time.time() * 1000)
    gap_minutes = (current_ts - last_ts) / 60000
    print(f"[SYNC] Gap da colmare: {gap_minutes:.0f} minuti ({gap_minutes/60:.1f} ore)")

    new_data = []
    batches = 0

    while last_ts < current_ts - 300000:  # stop 5 min before now
        url = f"https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=5m&startTime={last_ts + 1}&limit=1000"
        try:
            response = requests.get(url, timeout=15)
            print(f"[SYNC] Binance HTTP status: {response.status_code}")
            res = response.json()
        except Exception as e:
            print(f"[SYNC] ERRORE richiesta Binance: {e}")
            sync_status["last_error"] = str(e)
            sync_status["last_result"] = "fetch_error"
            break

        # Check if Binance returned an error dict instead of a list
        if isinstance(res, dict):
            print(f"[SYNC] ERRORE Binance API: {res}")
            sync_status["last_error"] = str(res)
            sync_status["last_result"] = "binance_api_error"
            break

        if not res or len(res) == 0:
            print("[SYNC] Nessun dato restituito da Binance. Fine sync.")
            break

        for k in res:
            try:
                new_data.append((
                    datetime.fromtimestamp(k[0]/1000).isoformat(),
                    float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]), 'BTCUSDT'
                ))
            except Exception as e:
                print(f"[SYNC] Errore parsing candela: {e} - raw: {k}")

        last_ts = res[-1][0]
        batches += 1
        print(f"[SYNC] Batch {batches}: aggiornato fino al {datetime.fromtimestamp(last_ts/1000)} ({len(new_data)} candele totali)")

        if len(res) < 1000:
            print("[SYNC] Fine dati disponibili da Binance.")
            break
        time.sleep(0.5)

    if new_data:
        df_new = pd.DataFrame(new_data, columns=["Open time", "Open", "High", "Low", "Close", "Volume", "symbol"])
        df_new.to_sql("prices", conn, if_exists="append", index=False)
        print(f"[SYNC] ✅ Salvate {len(new_data)} nuove candele nel DB.")
        sync_status["candles_added_last_run"] = len(new_data)
        sync_status["last_result"] = "success"
        sync_status["last_error"] = None
    else:
        print("[SYNC] Nessuna nuova candela da salvare.")
        sync_status["candles_added_last_run"] = 0
        if sync_status["last_result"] not in ("fetch_error", "binance_api_error"):
            sync_status["last_result"] = "up_to_date"

    conn.close()

def auto_sync_loop():
    while True:
        try:
            print(f"[SYNC LOOP] Avvio sync alle {datetime.now().isoformat()}")
            sync_data()
        except Exception as e:
            print(f"[SYNC LOOP] Eccezione non gestita: {e}")
            sync_status["last_error"] = str(e)
            sync_status["last_result"] = "unhandled_exception"
        print(f"[SYNC LOOP] Prossima sync tra 10 minuti.")
        time.sleep(600)

@app.on_event("startup")
def startup_event():
    print("[STARTUP] Avvio thread di sincronizzazione Binance...")
    thread = threading.Thread(target=auto_sync_loop, daemon=True)
    thread.start()
    print("[STARTUP] Thread avviato.")

def get_prediction(strike_price, minutes_to_expiry):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT \"Open time\", Close FROM prices ORDER BY \"Open time\" DESC LIMIT 100000", conn)
    conn.close()
    
    if len(df) < minutes_to_expiry + 1000:
        return {"error": "Dati insufficienti per la predizione"}

    df = df.iloc[::-1].reset_index(drop=True)
    df['returns'] = df['Close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=60).std()
    
    current_vol = df['volatility'].iloc[-1]
    vol_quartiles = df['volatility'].quantile([0.25, 0.5, 0.75])
    
    if current_vol <= vol_quartiles[0.25]:
        regime = "Low Vol (Q1)"
        filtered_df = df[df['volatility'] <= vol_quartiles[0.25]]
    elif current_vol <= vol_quartiles[0.5]:
        regime = "Med-Low Vol (Q2)"
        filtered_df = df[(df['volatility'] > vol_quartiles[0.25]) & (df['volatility'] <= vol_quartiles[0.5])]
    elif current_vol <= vol_quartiles[0.75]:
        regime = "Med-High Vol (Q3)"
        filtered_df = df[(df['volatility'] > vol_quartiles[0.5]) & (df['volatility'] <= vol_quartiles[0.75])]
    else:
        regime = "High Vol (Q4)"
        filtered_df = df[df['volatility'] > vol_quartiles[0.75]]

    current_price = df['Close'].iloc[-1]
    target_return = (strike_price / current_price) - 1
    
    # Calcolo dei ritorni storici proiettati
    sample_returns = []
    for i in filtered_df.index:
        if i + minutes_to_expiry < len(df):
            fwd_return = (df['Close'].iloc[i + minutes_to_expiry] / df['Close'].iloc[i]) - 1
            sample_returns.append(fwd_return)
    
    if not sample_returns:
        return {"error": "Nessun campione storico trovato per questo regime"}
        
    hits = sum(1 for r in sample_returns if r >= target_return)
    probability = (hits / len(sample_returns)) * 100
    
    return {
        "probability": round(probability, 2),
        "regime": regime,
        "current_price": round(current_price, 2),
        "required_move": f"{round(target_return * 100, 2)}%",
        "samples": len(sample_returns)
    }

@app.get("/predict")
def predict(strike: float, expiry: int):
    return get_prediction(strike, expiry)

@app.get("/sync-status")
def get_sync_status():
    return {"sync": sync_status}

@app.get("/status")
def status():
    try:
        conn = sqlite3.connect(DB_PATH)
        count = pd.read_sql("SELECT COUNT(*) as n FROM prices", conn).iloc[0,0]
        oldest = pd.read_sql('SELECT MIN("Open time") as t FROM prices', conn).iloc[0,0]
        newest = pd.read_sql('SELECT MAX("Open time") as t FROM prices', conn).iloc[0,0]
        conn.close()
        return {
            "status": "online",
            "timestamp": datetime.now().isoformat(),
            "database": {
                "candles": int(count),
                "oldest": oldest,
                "newest": newest,
                "interval": "5min"
            }
        }
    except Exception as e:
        return {"status": "online", "timestamp": datetime.now().isoformat(), "database": {"error": str(e)}}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
