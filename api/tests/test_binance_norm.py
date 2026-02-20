from api.binance_data import normalize_binance_symbol

test_symbols = [
    "ELF/USDT.BINANCE",
    "BTCUSDT.BINANCE",
    "ETH/USDT",
    "BINANCE:BNBUSDT",
    "ADA-USDT",
    "SOL_USDT",
    "DOTUSD",
]

print("Testing normalize_binance_symbol:")
for sym in test_symbols:
    normalized = normalize_binance_symbol(sym)
    print(f"{sym} -> {normalized}")
