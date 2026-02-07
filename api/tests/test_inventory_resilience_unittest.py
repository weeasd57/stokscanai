import unittest
from unittest.mock import patch, MagicMock
import httpx
import os
import sys

# Ensure api module is findable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import api.stock_ai as stock_ai

class TestInventoryResilience(unittest.TestCase):
    def setUp(self):
        # Save original state
        self.orig_supabase = stock_ai.supabase
        self.orig_init = stock_ai._init_supabase

    def tearDown(self):
        # Restore original state
        stock_ai.supabase = self.orig_supabase
        stock_ai._init_supabase = self.orig_init

    def test_inventory_rpc_fails_fallback_to_fundamentals(self):
        """Test that if the get_inventory_stats RPC fails, it falls back to fundamentals table."""
        
        class MockQuery:
            def __init__(self, target):
                self.target = target
            def select(self, *args, **kwargs): return self
            def eq(self, *args, **kwargs): return self
            def order(self, *args, **kwargs): return self
            def limit(self, *args, **kwargs): return self
            def execute(self):
                if self.target == "rpc":
                    # Simulate a permanent failure for RPC (after retries)
                    raise httpx.ConnectError("RPC Failed")
                elif self.target == "stock_fundamentals":
                    # Mock data for fundamentals fallback
                    mock_res = MagicMock()
                    mock_res.data = [
                        {"exchange": "NYSE", "data": {"country": "USA"}},
                        {"exchange": "NYSE", "data": {"country": "USA"}},
                        {"exchange": "LSE", "data": {"country": "UK"}}
                    ]
                    return mock_res
                return MagicMock(data=[])

        class MockSupabase:
            def rpc(self, name, params=None):
                return MockQuery("rpc")
            def table(self, name):
                return MockQuery(name)

        fake_sb = MockSupabase()
        # We need to ensure stock_ai.supabase is set correctly
        # AND restored by _init_supabase during retries
        stock_ai.supabase = fake_sb
        stock_ai._init_supabase = lambda: setattr(stock_ai, 'supabase', fake_sb)

        # Mock symbols_local to avoid file I/O and non-existent files in test environment
        with patch("api.symbols_local.load_country_summary", return_value={}):
            # We want to force fast retries for tests
            original_read = stock_ai._supabase_read_with_retry
            
            def mock_read_with_retry(query_func_or_table, **kwargs):
                kwargs['sleep_base_s'] = 0.001
                kwargs['max_attempts'] = 2 # Speed up failure
                return original_read(query_func_or_table, **kwargs)
            
            with patch("api.stock_ai._supabase_read_with_retry", side_effect=mock_read_with_retry):
                inventory = stock_ai.get_supabase_inventory()
                
                # The RPC call happened and failed (retried twice)
                # Then the fundamentals call happened and succeeded
                
                exchanges = [i['exchange'] for i in inventory]
                self.assertIn("NYSE", exchanges)
                self.assertIn("LSE", exchanges)
                
                # Check fund_count (NYSE had 2 rows, LSE had 1)
                nyse = next(i for i in inventory if i['exchange'] == "NYSE")
                self.assertEqual(nyse['fund_count'], 2)
                # price_count should be 0 because we couldn't determine it in fallback mode
                self.assertEqual(nyse['price_count'], 0)

    def test_inventory_transient_rpc_failure_retries_and_succeeds(self):
        """Test that a transient RPC failure eventually succeeds via retry."""
        
        shared_state = {"rpc_calls": 0}

        class MockQuery:
            def execute(self):
                shared_state["rpc_calls"] += 1
                if shared_state["rpc_calls"] == 1:
                    raise httpx.RemoteProtocolError("EOF")
                # Succeed on second call
                mock_res = MagicMock()
                mock_res.data = [{"exchange": "EGX", "fund_count": 10, "price_count": 5}]
                return mock_res

        class MockSupabase:
            def rpc(self, name, params=None): return MockQuery()
            def table(self, name): return MagicMock()

        fake_sb = MockSupabase()
        stock_ai.supabase = fake_sb
        stock_ai._init_supabase = lambda: setattr(stock_ai, 'supabase', fake_sb)

        original_read = stock_ai._supabase_read_with_retry
        def mock_read_with_retry(query_func_or_table, **kwargs):
            kwargs['sleep_base_s'] = 0.001
            return original_read(query_func_or_table, **kwargs)

        with patch("api.stock_ai._supabase_read_with_retry", side_effect=mock_read_with_retry):
            # We also mock fundamentals just in case it leaks
            with patch("api.symbols_local.load_country_summary", return_value={}):
                inventory = stock_ai.get_supabase_inventory()
                
                # Should have called RPC twice
                self.assertEqual(shared_state["rpc_calls"], 2)
                
                egx = next(i for i in inventory if i['exchange'] == "EGX")
                self.assertEqual(egx['fund_count'], 10)
                self.assertEqual(egx['price_count'], 5)

if __name__ == "__main__":
    unittest.main()
