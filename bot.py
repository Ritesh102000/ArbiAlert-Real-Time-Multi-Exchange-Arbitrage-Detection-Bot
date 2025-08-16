import os
import asyncio
import logging
import nest_asyncio
nest_asyncio.apply()

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import (
    ApplicationBuilder, CommandHandler, CallbackQueryHandler,
    MessageHandler, ConversationHandler, ContextTypes, filters
)
import pandas as pd
import warnings
import json

# === Setup Directories ===
USER_CONFIG_DIR = "UserConfig"
USER_SEEN_DIR = "UserSeen"
os.makedirs(USER_CONFIG_DIR, exist_ok=True)
os.makedirs(USER_SEEN_DIR, exist_ok=True)

# === Logging Setup ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler()
    ]
)

# === Imports ===
from config import symbol_to_exchanges, symbol_to_exchanges_cbbo
# === Constants ===
SELECT_SYMBOL, SELECT_EXCHANGES, ENTER_THRESHOLD, SELECT_SYMBOLS = range(4)
BOT_TOKEN = "7320070279:AAEWXcZUpu0nwVX3VY1bSRDmnImpVJr-bLg"
# === Helpers ===
def log_user(update, command):
    user = update.effective_user
    logging.info(f"Command: {command} by User: {user.id} - @{user.username or ''}")

def log_message_sent(user_id, message):
    logging.info(f"Sent to {user_id}: {message}")



def load_user_seen(user_id):
    path = os.path.join(USER_SEEN_DIR, f"{user_id}.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return set(tuple(x) for x in json.load(f))
    return set()

def save_user_seen(user_id, seen_set):
    path = os.path.join(USER_SEEN_DIR, f"{user_id}.json")
    with open(path, "w") as f:
        json.dump([list(k) for k in seen_set], f)


# Setup logging
logging.basicConfig(level=logging.INFO)


# Handlers
# === Command Handlers ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    logging.info(f"User ID: {user.id}, Name: {user.first_name} {user.last_name or ''}, Username: @{user.username or ''}")
    await update.message.reply_text(
        "ℹ️ *Available Commands:*\n"
        "/monitor_arb - Start monitoring arbitrage\n"
        "/cancel - Cancel the current action",
        
    )

async def help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "ℹ️ *Available Commands:*\n"
        "/monitor_arb - Start monitoring arbitrage\n"
        "/select_symbols - Select symbols to monitor CBBO (e.g. BTCUSDT, ETHUSDT)\n"
        "/status - Show current monitored symbols\n"
        "/cancel - Cancel the current action\n"
        "/force_check - Force send latest arbitrage update",
        
    )


async def monitor_arb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Always clear previous user data
    context.user_data.clear()

    # If it's a message (user typed /monitor_arb)
    if update.message:
        
        keyboard = [[InlineKeyboardButton(symbol, callback_data=symbol)] for symbol in symbol_to_exchanges]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text("📈 Select a symbol to monitor:", reply_markup=reply_markup, parse_mode=None)
        return SELECT_SYMBOL

    # If it's a callback (called internally from inside the convo)
    elif update.callback_query:
        query = update.callback_query
        await query.edit_message_text("📈 Restarting...\nSelect a symbol to monitor:", parse_mode=None)
        keyboard = [[InlineKeyboardButton(symbol, callback_data=symbol)] for symbol in symbol_to_exchanges]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.message.reply_text("📈 Select a symbol to monitor:", reply_markup=reply_markup, parse_mode=None)
        return SELECT_SYMBOL


async def symbol_selected(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    selected_symbol = query.data
    context.user_data['symbol'] = selected_symbol
    context.user_data['selected_exchanges'] = []

    return await send_exchange_selection_menu(query, context)


async def send_exchange_selection_menu(query, context):
    exchanges = symbol_to_exchanges[context.user_data['symbol']]
    selected = context.user_data['selected_exchanges']

    keyboard = [
        [InlineKeyboardButton(f"{'✅ ' if ex in selected else ''}{ex}", callback_data=ex)]
        for ex in exchanges
    ] + [[InlineKeyboardButton("✅ Done", callback_data="done")]]
    
    await query.edit_message_text(
        text=f"🧭 Select exchange(s) to monitor (press ✅ when done):\nSelected: {', '.join(selected) or 'None'}",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return SELECT_EXCHANGES


async def exchange_selected(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    exchange = query.data
    selected = context.user_data['selected_exchanges']

    if exchange == "done":
        if len(selected) < 2:
            await query.edit_message_text("⚠️ You must select at least *2 exchanges* to proceed.\nRestarting...", parse_mode="Markdown")
            # Clear selections and restart from symbol selection
            return await monitor_arb(update, context)
        
        await query.edit_message_text("💬 Now, please enter a threshold value (e.g., 0.001):")
        return ENTER_THRESHOLD

    # Toggle logic with limit check
    if exchange in selected:
        selected.remove(exchange)
    else:
        selected.append(exchange)

    return await send_exchange_selection_menu(query, context)


async def threshold_entered(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    try:
        threshold = float(text)

        symbol = context.user_data['symbol']
        exchanges = context.user_data['selected_exchanges']
        user_id = update.effective_user.id

        # Build data rows
        rows = []
        for i in exchanges:
            rows.append({
                "path": f"orderbook_data/{i}_{symbol}-USDT",
                "threshold": threshold,
                "symbol": symbol,
                "exchange": i
            })
            rows.append({
                "path": f"orderbook_data/{i}_{symbol}-BTC",
                "threshold": threshold,
                "symbol": symbol,
                "exchange": i
            })

        # Save to user-specific CSV
        os.makedirs("UserState", exist_ok=True)
        filename = f"UserState/{user_id}.csv"
        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)

        await update.message.reply_text(
            f"✅ Monitoring started!\n"
            f"Symbol: `{symbol}`\n"
            f"Exchanges: `{', '.join(exchanges)}`\n"
            f"Threshold: `{threshold}`",
            parse_mode="Markdown"
        )

        # Optional: Start background monitoring
        # asyncio.create_task(your_monitor_function(...))

        return ConversationHandler.END

    except ValueError:
        await update.message.reply_text(
            "❌ Invalid input. Please enter a *number* (e.g., 0.001 or 1.5).",
            parse_mode="Markdown"
        )
        return ENTER_THRESHOLD


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("❌ Monitoring cancelled.")
    return ConversationHandler.END


async def timeout(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("⌛ Timed out. Please start again with /monitor_arb.")
    return ConversationHandler.END
from telegram.constants import ParseMode
SELECT_SYMBOLS = 100  # Use a unique number to avoid collision

# /select_symbols command entry point
async def select_symbols(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = []
    for symbol in symbol_to_exchanges_cbbo.keys():
        keyboard.append([InlineKeyboardButton(symbol, callback_data=symbol)])
    keyboard.append([InlineKeyboardButton("✅ Done", callback_data="done")])

    context.user_data["selected_symbols"] = set()

    await update.message.reply_text(
        "📊 *Select the symbols you want to monitor.*\nPress ✅ Done when finished.",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode=ParseMode.MARKDOWN
    )
    return SELECT_SYMBOLS


# Callback handler for symbol selection
async def handle_symbol_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = update.effective_user.id
    symbol = query.data

    if symbol == "done":
        selected = list(context.user_data["selected_symbols"])
        selected = [f"orderbook_data\\{i}_CBBO.csv" for i in selected]
        if not selected:
            await query.edit_message_text("⚠️ No symbols selected. Try again with /select_symbols.")
            return ConversationHandler.END

        # Save to config
        config_path = os.path.join(USER_CONFIG_DIR, f"{user_id}.json")
        with open(config_path, "w") as f:
            json.dump({"path": selected}, f, indent=2)

        await query.edit_message_text(
            f"✅ Monitoring saved for user `{user_id}`.\nSymbols: {', '.join(selected)}",
            parse_mode=ParseMode.MARKDOWN
        )
        return ConversationHandler.END

    # Toggle symbol selection
    selected = context.user_data["selected_symbols"]
    if symbol in selected:
        selected.remove(symbol)
    else:
        selected.add(symbol)

    # Update UI with selected symbols
    keyboard = []
    for sym in symbol_to_exchanges_cbbo.keys():
        label = f"✔️ {sym}" if sym in selected else sym
        keyboard.append([InlineKeyboardButton(label, callback_data=sym)])
    keyboard.append([InlineKeyboardButton("✅ Done", callback_data="done")])

    await query.edit_message_reply_markup(reply_markup=InlineKeyboardMarkup(keyboard))
    return SELECT_SYMBOLS
async def monitor_user_signals(app):
    await asyncio.sleep(5)  # Small delay to let bot fully initialize

    while True:
        for filename in os.listdir(USER_CONFIG_DIR):
            if not filename.endswith(".json"):
                continue

            user_id = filename.replace(".json", "")
            config_path = os.path.join(USER_CONFIG_DIR, filename)

            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                csv_paths = config.get("path", [])

                for path in csv_paths:
                    if not os.path.exists(path):
                        continue

                    # Read only last row
                    try:
                        df = pd.read_csv(path)
                        if df.empty:
                            continue

                        last = df.iloc[-1]
                        symbol = last['symbol']
                        bid = last['cbbo_bid']
                        bid_ex = last['cbbo_bid_ex']
                        ask = last['cbbo_ask']
                        ask_ex = last['cbbo_ask_ex']
                        mid = last['mid_price']

                        msg = (
                            f"{symbol}: Best Bid on {bid_ex.capitalize()} @ ${bid:.2f}, "
                            f"Best Offer on {ask_ex.capitalize()} @ ${ask:.2f}. "
                            f"CBBO Mid: ${mid:.2f}"
                        )

                        # Send message to user
                        try:
                            await app.bot.send_message(chat_id=user_id, text=msg)
                        except Exception as send_err:
                            logging.warning(f"Failed to send to user {user_id}: {send_err}")

                    except Exception as read_err:
                        logging.warning(f"Failed to read CSV {path}: {read_err}")

            except Exception as user_err:
                logging.warning(f"Failed to process user config {filename}: {user_err}")

        await asyncio.sleep(100)  # Run every 10 seconds
# Conversation Handler
conv_handler = ConversationHandler(
    entry_points=[CommandHandler("monitor_arb", monitor_arb),CommandHandler("select_symbols", select_symbols)],
    states={
        SELECT_SYMBOL: [CallbackQueryHandler(symbol_selected)],
        SELECT_EXCHANGES: [CallbackQueryHandler(exchange_selected)],
        ENTER_THRESHOLD: [MessageHandler(filters.TEXT & ~filters.COMMAND, threshold_entered)],
        SELECT_SYMBOLS: [CallbackQueryHandler(handle_symbol_selection)],
    },
    fallbacks=[
        CommandHandler("cancel", cancel),
        CommandHandler("monitor_arb", monitor_arb),
         # ✅ Add this line
    ],
    conversation_timeout=60
)
def load_user_configurations():
    configs = []
    for filename in os.listdir("UserState"):
        if not filename.endswith(".csv"):
            continue
        user_id = int(filename.replace(".csv", ""))
        df = pd.read_csv(os.path.join("UserState", filename))
        if df.empty:
            continue
        for symbol in df["symbol"]:
            rows = df[df["symbol"] == symbol]
            threshold = rows["threshold"].iloc[0]
            paths = rows["path"].tolist()
            configs.append({
                "user_id": user_id,
                "symbol": symbol,
                "paths": paths,
                "threshold": threshold
            })
    return configs


from itertools import combinations
def detect_arbitrage_from_files(csv_files, btc_usdt_price, threshold=0.00001, user_id=None):
    def load_latest_row(csv_path):
        df = pd.read_csv(csv_path)
        latest = df.iloc[-1]
        latest['timestamp'] = pd.to_datetime(latest['timestamp'])
        latest['minute'] = latest['timestamp'].floor('min')
        return latest

    def convert_to_usdt(row, btc_usdt):
        symbol = row['symbol']
        if symbol.endswith('BTC'):
            row['best_bid'] = float(row['best_bid']) * btc_usdt
            row['best_ask'] = float(row['best_ask']) * btc_usdt
            
        else:
            row['best_bid'] = float(row['best_bid'])
            row['best_ask'] = float(row['best_ask'])
        return row

    def log_arbitrage(symbol, timestamp, buy_exchange, sell_exchange, buy_price, sell_price, spread):
        threshold_formatted = f"{threshold:.4f}".replace('.', '_')
        os.makedirs("latest_arbitrage", exist_ok=True)
        log_file = f"latest_arbitrage/arbitrage_{threshold_formatted}_{symbol}.csv"
        log_entry = {
            "timestamp": timestamp,
            "symbol": symbol,
            "buy_exchange": buy_exchange,
            "sell_exchange": sell_exchange,
            "buy_price": buy_price,
            "sell_price": sell_price,
            "spread": spread
        }

        df_entry = pd.DataFrame([log_entry])
        if os.path.exists(log_file):
            existing = pd.read_csv(log_file)
            if not ((existing['timestamp'] == timestamp) & 
                    (existing['buy_exchange'] == buy_exchange) & 
                    (existing['sell_exchange'] == sell_exchange)).any():
                df_entry.to_csv(log_file, mode='a', header=False, index=False)
        else:
            df_entry.to_csv(log_file, mode='w', header=True, index=False)

    if user_id is None:
        return []

    seen_opportunities = load_user_seen(user_id)
    latest_data = []

    for path in csv_files:
        if not os.path.exists(path):
            print(f"❌ File not found: {path}")
            continue
        try:
            
            row = load_latest_row(path)
            
            row = convert_to_usdt(row, btc_usdt_price)
            latest_data.append(row)
        except Exception as e:
            print(f"⚠️ Error reading {path}: {e}")

    messages = []

    if not latest_data:
        return messages

    df_all = pd.DataFrame(latest_data)

    for symbol, group in df_all.groupby('symbol'):
        for row1, row2 in combinations(group.itertuples(), 2):
            
            if row1.minute != row2.minute:
                continue
            print(f"🔍 Comparing {row1.symbol} on {row1.exchange} and {row2.exchange} at {row1.minute}")
            ask1, bid1 = row1.best_ask, row1.best_bid
            ask2, bid2 = row2.best_ask, row2.best_bid
            timestamp = str(row1.minute)

            # Case 1: buy on row1, sell on row2
            if bid2 - ask1 > threshold:
                spread = bid2 - ask1
                key = (symbol, row1.exchange, row2.exchange, timestamp)
                if key not in seen_opportunities:
                    seen_opportunities.add(key)
                    log_arbitrage(symbol, timestamp, row1.exchange, row2.exchange, ask1, bid2, spread)
                    msg = (f"🟢 Arbitrage Opportunity:\n"
                           f"Symbol: *{symbol}*\n"
                           f"Buy on *{row1.exchange}* at `{ask1:.4f}`\n"
                           f"Sell on *{row2.exchange}* at `{bid2:.4f}`\n"
                           f"Spread: `{spread:.6f}`")
                    messages.append(msg)

            # Case 2: buy on row2, sell on row1
            elif bid1 - ask2 > threshold:
                spread = bid1 - ask2
                key = (symbol, row2.exchange, row1.exchange, timestamp)
                if key not in seen_opportunities:
                    seen_opportunities.add(key)
                    log_arbitrage(symbol, timestamp, row2.exchange, row1.exchange, ask2, bid1, spread)
                    msg = (f"🟢 Arbitrage Opportunity:\n"
                           f"Symbol: *{symbol}*\n"
                           f"Buy on *{row2.exchange}* at `{ask2:.4f}`\n"
                           f"Sell on *{row1.exchange}* at `{bid1:.4f}`\n"
                           f"Spread: `{spread:.6f}`")
                    messages.append(msg)
        # 🔄 Compare BTC and USDT pairs on the same exchange
    exchange_groups = df_all.groupby('exchange')

    for exchange, group in exchange_groups:
        symbols = group['symbol'].tolist()
        
        btc_row = group[group['symbol'].str.endswith('BTC')]
        usdt_row = group[group['symbol'].str.endswith('USDT')]

        if btc_row.empty or usdt_row.empty:
            continue

        btc_row = btc_row.iloc[0]
        for _, usdt_row in usdt_row.iterrows():
            timestamp = str(btc_row['minute'])
            symbol_usdt = usdt_row['symbol']
            
            # Case 1: Buy BTC, Sell USDT pair (both on same exchange)
            if usdt_row['best_bid'] - btc_row['best_ask'] > threshold:
                spread = usdt_row['best_bid'] - btc_row['best_ask']
                key = (symbol_usdt + "_BTC", exchange, exchange, timestamp)
                if key not in seen_opportunities:
                    seen_opportunities.add(key)
                    log_arbitrage(symbol_usdt + "_BTC", timestamp, exchange, exchange, btc_row['best_ask'], usdt_row['best_bid'], spread)
                    msg = (f"🟢 *Same Exchange Arbitrage Opportunity*:\n"
                           f"Exchange: *{exchange}*\n"
                           f"Buy *BTC/USDT* at `{btc_row['best_ask']:.4f}`\n"
                           f"Sell *{symbol_usdt}* at `{usdt_row['best_bid']:.4f}`\n"
                           f"Spread: `{spread:.6f}`")
                    messages.append(msg)

            # Case 2: Buy USDT pair, Sell BTC (both on same exchange)
            elif btc_row['best_bid'] - usdt_row['best_ask'] > threshold:
                spread = btc_row['best_bid'] - usdt_row['best_ask']
                key = (symbol_usdt + "_USDT", exchange, exchange, timestamp)
                if key not in seen_opportunities:
                    seen_opportunities.add(key)
                    log_arbitrage(symbol_usdt + "_USDT", timestamp, exchange, exchange, usdt_row['best_ask'], btc_row['best_bid'], spread)
                    msg = (f"🟢 *Same Exchange Arbitrage Opportunity*:\n"
                           f"Exchange: *{exchange}*\n"
                           f"Buy *{symbol_usdt}* at `{usdt_row['best_ask']:.4f}`\n"
                           f"Sell *BTC/USDT* at `{btc_row['best_bid']:.4f}`\n"
                           f"Spread: `{spread:.6f}`")
                    messages.append(msg)


    save_user_seen(user_id, seen_opportunities)
    print(messages)
    return set(messages)

async def monitor_all_users(application):
    while True:
        logging.info("🔁 Checking arbitrage for all users...")

        configs = load_user_configurations()
        btc_usdt_price = 107417.06  # Replace with real-time price fetch if needed

        for config in configs:
            try:
                symbol = config["symbol"]
                files = [os.path.join(f"{path}.csv") for path in config["paths"]]

                # Pass user_id explicitly to avoid shared seen set
                logs = detect_arbitrage_from_files(
                    files, btc_usdt_price,
                    threshold=config.get("threshold", 0.00001),
                    user_id=str(config["user_id"])  # Ensure it's a string (for filenames)
                )

                if logs:
                    message = f"📢 Arbitrage for *{symbol}*:\n" + "\n\n".join(logs)
                    await application.bot.send_message(
                        chat_id=config["user_id"],
                        text=message,
                        parse_mode="Markdown"
                    )

            except Exception as e:
                logging.error(f"⚠️ Error monitoring user {config['user_id']}: {e}")

        await asyncio.sleep(10)  # Adjust as needed (e.g. 300 for 5 minutes)

async def force_check(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔁 Running arbitrage check now...")
    await monitor_all_users(context.application)

# Add this:
async def user_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    config_path = f"UserState/{user_id}.csv"
    seen_path = f"UserSeen/{user_id}_seen.json"

    # 1. Check if config exists
    if not os.path.exists(config_path):
        await update.message.reply_text(
            "⚠️ You don't have any arbitrage alert config yet.\n"
            "Please register first to start receiving alerts."
        )
        return

    # 2. Check if user has received any opportunities
    if not os.path.exists(seen_path):
        await update.message.reply_text("✅ You are registered but no arbitrage opportunities have been found for you yet. Please wait a bit longer!")
        return

    # 3. Return last seen opportunity
    try:
        with open(seen_path, "r") as f:
            data = json.load(f)
            if not data:
                await update.message.reply_text("✅ You are registered but no arbitrage opportunities have been found for you yet. Please wait a bit longer!")
                return
            last_opportunity = data[-1]  # Assuming chronological order

        # Format the response
        message = (
            "📊 *Your Last Arbitrage Opportunity:*\n"
            f"🔁 `{last_opportunity[0]}`\n"
            f"💰 `{last_opportunity[1]}` → `{last_opportunity[2]}`\n"
            f"📈 Profit: `{last_opportunity[3]*100:.2f}%`"
        )
        await update.message.reply_text(message, parse_mode="Markdown")

    except Exception as e:
        await update.message.reply_text("❌ Something went wrong while retrieving your alert.")
        logging.error(f"Error in /status for user {user_id}: {e}")
import aiohttp

async def unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await help(update, context)
async def fetch_symbols(exchange: str):
    url = f"https://gomarket-api.goquant.io/api/symbols/{exchange.lower()}/spot"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                return [item["name"] for item in data.get("symbols", [])]
            else:
                return []
def get_symbol_page(symbols, page, page_size=30):
    total_pages = (len(symbols) - 1) // page_size + 1
    start = page * page_size
    end = start + page_size
    page_symbols = symbols[start:end]
    return page_symbols, total_pages

from telegram.ext import CallbackQueryHandler
exchanges = ['okx', 'deribit', 'bybit', 'binance', 'cryptocom', 'kraken', 'kucoin', 'bitstamp', 'bitmex', 'coinbase_intl', 'coinbase', 'bitfinex', 'gateio', 'mexc', 'gemini', 'htx', 'bitget', 'dydx', 'bitso']
async def symbols_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton(exchange.upper(), callback_data=f"symbols_{exchange.upper()}")]
        for exchange in exchanges
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Select an exchange to view symbols:", reply_markup=reply_markup)

async def symbol_exchange_selected(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    _, exchange = query.data.split("_")
    symbols = await fetch_symbols(exchange.lower())

    if not symbols:
        await query.edit_message_text("❌ Could not fetch symbols.")
        return

    context.user_data["symbols"] = symbols
    context.user_data["exchange"] = exchange
    context.user_data["page"] = 0

    page_symbols, total_pages = get_symbol_page(symbols, 0)
    keyboard = []

    if total_pages > 1:
        keyboard = [[InlineKeyboardButton("⏭️ Next", callback_data="next_page")]]

    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(
        f"📊 Symbols on {exchange.upper()} (Page 1 of {total_pages}):\n\n" + "\n".join(page_symbols),
        reply_markup=reply_markup
    )
async def paginate_symbols(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    direction = query.data  # "next_page" or "prev_page"

    symbols = context.user_data.get("symbols", [])
    exchange = context.user_data.get("exchange", "")
    page = context.user_data.get("page", 0)

    total_pages = (len(symbols) - 1) // 30 + 1

    if direction == "next_page" and page < total_pages - 1:
        page += 1
    elif direction == "prev_page" and page > 0:
        page -= 1

    context.user_data["page"] = page
    page_symbols, _ = get_symbol_page(symbols, page)

    keyboard = []
    if page > 0:
        keyboard.append(InlineKeyboardButton("⏮️ Prev", callback_data="prev_page"))
    if page < total_pages - 1:
        keyboard.append(InlineKeyboardButton("⏭️ Next", callback_data="next_page"))

    reply_markup = InlineKeyboardMarkup([keyboard] if keyboard else [])

    await query.edit_message_text(
        f"📊 Symbols on {exchange.upper()} (Page {page+1} of {total_pages}):\n\n" + "\n".join(page_symbols),
        reply_markup=reply_markup
    )

async def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(conv_handler)
    app.add_handler(CommandHandler("status", user_status))
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help))
    app.add_handler(CommandHandler("ex_symbols", symbols_command))
    app.add_handler(MessageHandler(filters.COMMAND, unknown_command))
    app.add_handler(CallbackQueryHandler(symbol_exchange_selected, pattern="^symbols_"))
    app.add_handler(CallbackQueryHandler(paginate_symbols, pattern="^(next_page|prev_page)$"))
    app.add_handler(CommandHandler("force_check", force_check))
    asyncio.create_task(monitor_all_users(app))
    asyncio.create_task(monitor_user_signals(app))
    logging.info("🤖 Bot is running...")
    await app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
