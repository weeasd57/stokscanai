import sys
import os

# Mock the Alpaca API and other dependencies to test the safety logic
class MockAPI:
    def list_positions(self):
        return []
    def submit_order(self, **kwargs):
        print(f"DEBUG: Order submitted with args: {kwargs}")
        return type('Order', (), {'id': 'test_order_id'})

def _normalize_symbol(symbol: str) -> str:
    return (symbol or "").strip().upper().replace("/", "")

def test_qty_safety():
    # Test cases: (qty, is_crypto, expected_safe_qty, should_submit)
    test_cases = [
        (1.0, True, 0.9999, True),
        (2e-09, True, 1e-08, True), # My new floor is 1e-08
        (1e-09, True, 1e-08, True), # Pushed to floor
        (5e-10, True, 0, False),    # Below floor, shouldn't submit
        (1.0, False, 1.0, True),    # Not crypto
    ]
    
    ALPACA_MIN_QTY = 1e-08
    
    for qty, is_crypto, expected, should_submit in test_cases:
        safe_qty = float(qty)
        if is_crypto:
            reduced_qty = safe_qty * 0.9999
            if reduced_qty < ALPACA_MIN_QTY and safe_qty >= ALPACA_MIN_QTY:
                safe_qty = ALPACA_MIN_QTY
            else:
                safe_qty = reduced_qty
            
            if safe_qty < ALPACA_MIN_QTY:
                print(f"SKIPPING: {qty} (is_crypto={is_crypto}) -> Result: Skip (Correct={not should_submit})")
                continue
        
        print(f"SUBMITTING: {qty} (is_crypto={is_crypto}) -> Result: {safe_qty} (Correct={abs(safe_qty - expected) < 1e-12})")

if __name__ == "__main__":
    test_qty_safety()
