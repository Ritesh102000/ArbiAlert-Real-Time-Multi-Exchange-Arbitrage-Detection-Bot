# ArbiAlert-Real-Time-Multi-Exchange-Arbitrage-Detection-Bot
Built a Telegram bot that polls multiple exchange APIs every second to fetch live market data for various assets and currency pairs. Implemented algorithms to detect arbitrage opportunities by comparing lowest bid and highest ask prices across platforms. 
# 🚀 Crypto Arbitrage Telegram Bot

## 📌 Project Overview
This project implements a **real-time arbitrage detection system** integrated with **Telegram** for instant user notifications.  
It continuously polls multiple **cryptocurrency exchange APIs** to fetch live orderbook data, processes it to detect **cross-exchange price discrepancies**, and alerts users about profitable opportunities.

---

## ⚙️ Key Features

### 🔄 Real-Time Market Data
- Polls **multiple exchange REST/WebSocket APIs** (e.g., OKX, Binance, Deribit, etc.) every second.  
- Fetches **Level-1/Level-2 orderbook data** (`best bid`, `best ask`, and depth).  
- Handles **API rate limits, retries, and latency** with asynchronous event loops (`asyncio`).  

### 📊 Arbitrage Opportunity Detection
- Implements **cross-exchange best bid-offer (CBBO)** calculations.  
- Compares:
  - **Highest Bid vs Lowest Ask** across exchanges.  
  - Same-asset **triangular arbitrage spreads** (optional extension).  
- Calculates **spread %** and checks against user-defined thresholds.  

### 👤 User Personalization
- Users can configure:
  - Assets (`ETH`, `BTC`, `SOL`, etc.)  
  - Trading pairs (`USDT`, `BTC`, `USD`, etc.)  
  - Exchanges to track  
  - Minimum arbitrage threshold (%) for alerts  
- Uses **persistent JSON/SQLite storage** to save preferences across sessions.  

### 📡 Instant Telegram Notifications
- Built with **`python-telegram-bot`** framework.  
- Sends structured alerts with:
  - Asset and pair  
  - Exchanges involved  
  - Bid/Ask prices and spread %  
  - Timestamp  
- Inline keyboard for quick user actions (change settings, view configs).  

### ⚡ Advanced Features
- **Latency Handling**: Adjusts for exchange response delays with timestamp validation.  
- **Asynchronous Processing**: Parallel API requests via `asyncio` + `aiohttp`.  
- **Multi-User Support**: Each user’s settings stored separately.  
- **Scalable Architecture**: Modular design for adding new exchanges easily.  

---

## 🛠️ Tech Stack

- **Python 3.10+**  
- **Asyncio** for concurrent non-blocking requests  
- **Aiohttp / HTTPX** for efficient API polling  
- **Python-Telegram-Bot** for Telegram integration  
- **Pandas** for spread calculation & data handling  
- **SQLite/JSON** for persistent storage  
- **Logging** for monitoring system health and debugging  

---

## 🔍 Example Workflow

1. User starts bot → selects assets (`ETH/USDT`, `ETH/BTC`) and threshold (`>0.5%`).  
2. Bot continuously polls exchange APIs asynchronously.  
3. For each asset:
   - Collects **best bid/ask** from all exchanges.  
   - Computes **arbitrage spreads** between lowest ask & highest bid.  
   - If spread ≥ threshold → sends **Telegram alert**.  
4. User receives message:  

💰 Arbitrage Opportunity Detected!
Asset: ETH/USDT
Buy @ OKX: 2500.10
Sell @ Binance: 2512.30
Spread: +0.49%

---

## 📈 Impact
- Enables **instant detection** of arbitrage opportunities without manual monitoring.  
- Provides **customizable trading insights** with multi-user support.  
- Designed for **scalability and extensibility**, supporting new assets and exchanges.  

---
