import os
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.virtual_market_adapter import VirtualMarketAdapter


def test_close_position_return_value():
    vm = VirtualMarketAdapter(price_provider=lambda s: 100.0)

    # No position yet
    assert vm.close_position("BTC/USD") is False

    # Buy to open
    vm.submit_order(symbol="BTC/USD", notional=1000.0, side="buy", type="market", time_in_force="gtc")
    assert len(vm.list_positions()) == 1

    # Close should return True and remove position
    assert vm.close_position("BTC/USD") is True
    assert len(vm.list_positions()) == 0

    # Closing again returns False
    assert vm.close_position("BTC/USD") is False


def test_seed_positions_from_state_hydrates_positions():
    vm = VirtualMarketAdapter(price_provider=lambda s: 10.0)
    vm.seed_positions_from_state(
        {
            "BTCUSD": {"symbol": "BTC/USD", "entry_price": 10, "amount": 2},
            "ETHUSD": {"entry_price": 10, "amount": 1},  # symbol missing -> inferred
        }
    )
    syms = sorted([p.symbol for p in vm.list_positions()])
    assert "BTC/USD" in syms
    assert "ETH/USD" in syms
    assert vm.close_position("BTC/USD") is True
    syms2 = sorted([p.symbol for p in vm.list_positions()])
    assert "ETH/USD" in syms2
