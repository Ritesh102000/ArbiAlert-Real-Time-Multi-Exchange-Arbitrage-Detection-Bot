import asyncio
import websockets
import json
import pandas as pd
from datetime import datetime
import os
from config import assets  # List of (exchange, symbol) tuples

DATA_FOLDER = "orderbook_data"
os.makedirs(DATA_FOLDER, exist_ok=True)

async def listen_orderbook(exchange, symbol):
    uri = f"wss://gomarket-api.goquant.io/ws/l1-orderbook/{exchange}/{symbol}"
    filename = os.path.join(DATA_FOLDER, f"{exchange}_{symbol.replace('/', '-').replace('_', '-')}.csv")
    print(f"Connecting to {uri}")

    while True:
        try:
            async with websockets.connect(uri) as websocket:
                print(f"[✓] Connected: {exchange} {symbol}")
                latest_data = None

                async def receiver():
                    nonlocal latest_data
                    while True:
                        message = await websocket.recv()
                        data = json.loads(message)
                        data["timestamp"] = datetime.utcnow().isoformat()
                        latest_data = data  # Overwrite with latest data

                # Background receiver task
                asyncio.create_task(receiver())

                # Save every 10 seconds
                while True:
                    await asyncio.sleep(10)

                    if latest_data:
                        clean_symbol = latest_data["symbol"].replace("_", "").replace("-", "").replace("/", "")
                        latest_data["symbol"] = clean_symbol
                        df = pd.DataFrame([latest_data])

                        if os.path.exists(filename):
                            df.to_csv(filename, mode='a', header=False, index=False)
                        else:
                            df.to_csv(filename, index=False)

        except Exception as e:
            print(f"[!] Error with {exchange} {symbol}: {e}. Reconnecting in 5 seconds...")
            await asyncio.sleep(5)

async def main():
    pairs_to_monitor = assets
    tasks = [asyncio.create_task(listen_orderbook(exchange, symbol)) for exchange, symbol in pairs_to_monitor]
    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user. Closing...")

if __name__ == "__main__":
    asyncio.run(main())
