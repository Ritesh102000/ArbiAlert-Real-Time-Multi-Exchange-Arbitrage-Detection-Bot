import asyncio
import websockets
import json
import pandas as pd
from datetime import datetime
import os
from collections import defaultdict
from config import assets  # List of (exchange, symbol) tuples

# === Setup ===
DATA_FOLDER = "orderbook_data"
os.makedirs(DATA_FOLDER, exist_ok=True)

# Symbol -> Exchange -> bid/ask
latest_quotes = defaultdict(dict)

# === Order Book Listener per Symbol/Exchange ===
async def listen_orderbook_l2(exchange, symbol):
    uri = f"wss://gomarket-api.goquant.io/ws/l2-orderbook/{exchange}/{symbol}"
    symbol_clean = symbol.replace("_", "").replace("-", "").replace("/", "")
    filename = os.path.join(DATA_FOLDER, f"{exchange}_{symbol_clean}_L2.csv")
    print(f"Connecting to: {uri}")

    try:
        async with websockets.connect(uri) as websocket:
            print(f"Connected to {exchange} {symbol}")
            latest_data = None

            async def receiver():
                nonlocal latest_data
                while True:
                    message = await websocket.recv()
                    data = json.loads(message)
                    data["timestamp"] = datetime.utcnow().isoformat()
                    latest_data = data

            asyncio.create_task(receiver())

            while True:
                await asyncio.sleep(5)

                if latest_data:
                    # Extract best bid and ask
                    bids = latest_data.get("bids", [])
                    asks = latest_data.get("asks", [])
                    best_bid = bids[0] if bids else [None, None]
                    best_ask = asks[0] if asks else [None, None]

                    flat = {
                        "timestamp": latest_data["timestamp"],
                        "exchange": exchange,
                        "symbol": symbol_clean,
                        "bid_price": best_bid[0],
                        "bid_size": best_bid[1],
                        "ask_price": best_ask[0],
                        "ask_size": best_ask[1],
                    }

                    # Save individual exchange quote
                    df = pd.DataFrame([flat])
                    if os.path.exists(filename):
                        df.to_csv(filename, mode='a', header=False, index=False)
                    else:
                        df.to_csv(filename, index=False)

                    # Update shared CBBO structure
                    latest_quotes[symbol_clean][exchange] = {
                        "bid": best_bid[0],
                        "ask": best_ask[0],
                        "timestamp": latest_data["timestamp"]
                    }

    except Exception as e:
        print(f"Connection error for {exchange} {symbol}: {e}")

# === CBBO Computation Task ===
async def compute_cbbo_loop():
    while True:
        await asyncio.sleep(10)
        
        for symbol, exchange_data in latest_quotes.items():
            best_bid = None
            best_ask = None
            best_bid_ex = ""
            best_ask_ex = ""

            for ex, data in exchange_data.items():
                bid = float(data.get("bid"))
                ask = float(data.get("ask"))

                if bid is not None and (best_bid is None or bid > best_bid):
                    best_bid = float(bid)
                    best_bid_ex = ex

                if ask is not None and (best_ask is None or ask < best_ask):
                    best_ask = float(ask)
                    best_ask_ex = ex

            if best_bid is not None and best_ask is not None:
                print(type(best_bid), best_ask, symbol)
                mid_price = (best_bid + best_ask) / 2
                timestamp = datetime.utcnow().isoformat()

                row = {
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "cbbo_bid": best_bid,
                    "cbbo_bid_ex": best_bid_ex,
                    "cbbo_ask": best_ask,
                    "cbbo_ask_ex": best_ask_ex,
                    "mid_price": mid_price
                }

                # Save per-symbol CBBO CSV
                cbbo_filename = os.path.join(DATA_FOLDER, f"{symbol}_CBBO.csv")
                df_cbbo = pd.DataFrame([row])
                if os.path.exists(cbbo_filename):
                    df_cbbo.to_csv(cbbo_filename, mode='a', header=False, index=False)
                else:
                    df_cbbo.to_csv(cbbo_filename, index=False)

# === Main Event Loop ===
async def main():
    pairs_to_monitor = assets
    tasks = [asyncio.create_task(listen_orderbook_l2(exchange, symbol)) for exchange, symbol in pairs_to_monitor]

    # Start CBBO calculation loop
    tasks.append(asyncio.create_task(compute_cbbo_loop()))

    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
