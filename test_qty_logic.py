
def simulate_sell_qty(qty, is_crypto=True):
    # Original logic
    safe_qty_orig = float(qty)
    if is_crypto:
        safe_qty_orig = safe_qty_orig * 0.9999
    
    # Proposed logic
    ALPACA_MIN_QTY = 1e-9
    safe_qty_new = float(qty)
    if is_crypto:
        # Only apply safety factor if it doesn't drop below min QTY
        # OR just use a more surgical approach
        reduced_qty = safe_qty_new * 0.9999
        if reduced_qty < ALPACA_MIN_QTY and safe_qty_new >= ALPACA_MIN_QTY:
            safe_qty_new = ALPACA_MIN_QTY # Keep at minimum
        else:
            safe_qty_new = reduced_qty

    return safe_qty_orig, safe_qty_new

def test():
    test_cases = [
        1e-9,      # Minimum allowed
        1.1e-9,    # Slightly above
        1e-8,      # Above
        0.5e-9,    # Below minimum (dust)
        1.0,       # Normal large qty
    ]
    
    print(f"{'Original Qty':<15} | {'Old Safe Qty':<15} | {'New Safe Qty':<15} | {'Status'}")
    print("-" * 65)
    for q in test_cases:
        old, new = simulate_sell_qty(q)
        status = "FIXED" if old < 1e-9 and new >= 1e-9 and q >= 1e-9 else "OK"
        if q < 1e-9:
            status = "DUST"
        print(f"{q:<15g} | {old:<15g} | {new:<15g} | {status}")

if __name__ == "__main__":
    test()
