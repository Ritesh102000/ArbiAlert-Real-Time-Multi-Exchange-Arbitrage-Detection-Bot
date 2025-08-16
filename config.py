assets = [
        ("okx", "ETH-USDT"),
        ("okx", "ETH-BTC"),
        ("deribit", "ETH_USDT"),
        ("deribit", "ETH_BTC"),
    ]
# Symbol to exchange mapping (could be from config.py)
symbol_to_exchanges = {
    "ETH": ["okx", "deribit"],
}
symbol_to_exchanges_cbbo = {
    "ETHBTC": ["okx", "deribit"],
    "ETHUSDT": ["okx", "deribit"],
}