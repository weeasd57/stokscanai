import os
import sys
import unittest

raise unittest.SkipTest("Alpaca routes removed (virtual execution only).")

import httpx
from fastapi.testclient import TestClient

# Add repo root to path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from api.main import app
import api.stock_ai as stock_ai


class _FakeQuery:
    def __init__(self, table_name: str, fail_count: int = 0):
        self._table_name = table_name
        self._fail_count = fail_count
        self._calls = 0

    def select(self, *args, **kwargs):
        return self

    def eq(self, *args, **kwargs):
        return self

    def in_(self, *args, **kwargs):
        return self

    def order(self, *args, **kwargs):
        return self

    def limit(self, *args, **kwargs):
        return self

    def execute(self):
        self._calls += 1
        if self._calls <= self._fail_count:
            raise httpx.RemoteProtocolError("Server disconnected")
        # Return plain object that mimics the expected structure
        class MockResponse:
            def __init__(self, data, count):
                self.data = data
                self.count = count
        return MockResponse(
            data=[{"ts": "2024-01-01T00:00:00Z", "date": "2024-01-01", "updated_at": "2024-01-01T00:00:00Z"}],
            count=42
        )


class _FakeSupabase:
    def __init__(self, fail_count: int = 0):
        self.fail_count = fail_count

    def table(self, table_name: str):
        return _FakeQuery(table_name, fail_count=self.fail_count)


class TestAlpacaSupabaseStats(unittest.TestCase):
    def test_transient_supabase_error_returns_503_after_all_retries(self):
        orig_supabase = stock_ai.supabase
        orig_init = stock_ai._init_supabase
        fake_sb = _FakeSupabase(fail_count=5)
        stock_ai._init_supabase = lambda: setattr(stock_ai, 'supabase', fake_sb)
        
        from unittest.mock import patch
        import api.routers.alpaca as alpaca_mod
        
        # Capture original to avoid recursion
        original_read = alpaca_mod._supabase_read_with_retry
        
        with patch.object(alpaca_mod, "_supabase_read_with_retry") as patched:
            patched.side_effect = lambda *a, **kw: original_read(*a, **{**kw, "sleep_base_s": 0.01})
            
            try:
                client = TestClient(app)
                resp = client.get("/alpaca/supabase-stats", params={"asset_class": "crypto"})
                self.assertEqual(resp.status_code, 503)
            finally:
                stock_ai.supabase = orig_supabase
                stock_ai._init_supabase = orig_init

    def test_transient_error_retries_and_succeeds(self):
        orig_supabase = stock_ai.supabase
        orig_init = stock_ai._init_supabase
        
        # We share the failure counter across all queries to ensure it clears
        shared_state = {"calls": 0, "fail_count": 2}
        
        class _SharedFakeQuery(_FakeQuery):
            def execute(self):
                shared_state["calls"] += 1
                if shared_state["calls"] <= shared_state["fail_count"]:
                    raise httpx.RemoteProtocolError("Server disconnected")
                return super().execute()

        class _SharedFakeSupabase(_FakeSupabase):
            def table(self, table_name: str):
                return _SharedFakeQuery(table_name)

        fake_sb = _SharedFakeSupabase()
        stock_ai._init_supabase = lambda: setattr(stock_ai, 'supabase', fake_sb)
        
        from unittest.mock import patch
        import api.routers.alpaca as alpaca_mod
        original_read = alpaca_mod._supabase_read_with_retry
        
        def mock_read_with_retry(*args, **kwargs):
            kwargs['sleep_base_s'] = 0.01
            kwargs['max_attempts'] = 5
            return original_read(*args, **kwargs)
            
        with patch.object(alpaca_mod, "_supabase_read_with_retry", side_effect=mock_read_with_retry) as patched:
            try:
                client = TestClient(app)
                resp = client.get("/alpaca/supabase-stats", params={"asset_class": "crypto"})
                if resp.status_code != 200:
                    print(f"TEST_FAILURE_BODY: {resp.text}")
                self.assertEqual(resp.status_code, 200)
                data = resp.json()
                self.assertEqual(data["asset_class"], "crypto")
            finally:
                stock_ai.supabase = orig_supabase
                stock_ai._init_supabase = orig_init


if __name__ == "__main__":
    unittest.main()

