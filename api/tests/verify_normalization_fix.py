from api.binance_data import normalize_binance_symbol

def test_normalization():
    test_cases = [
        ("BTC/USDT", "BTCUSDT"),
        ("ETH/USD", "ETHUSDT"),
        ("USDT/USD", "USDTUSD"), # The critical case
        ("BTCUSDT", "BTCUSDT"),
        ("USDTUSD", "USDTUSD"),
        ("BTC/EUR", "BTCEUR"),
        ("BINANCE:BTCUSDT", "BTCUSDT"),
    ]
    
    print(f"{'Input':<20} | {'Expected':<15} | {'Actual':<15} | {'Result'}")
    print("-" * 65)
    
    all_passed = True
    for inp, expected in test_cases:
        actual = normalize_binance_symbol(inp)
        passed = actual == expected
        print(f"{inp:<20} | {expected:<15} | {actual:<15} | {'✅ PASS' if passed else '❌ FAIL'}")
        if not passed:
            all_passed = False
            
    if all_passed:
        print("\nAll normalization tests passed!")
    else:
        print("\nSome normalization tests failed.")

if __name__ == "__main__":
    test_normalization()
