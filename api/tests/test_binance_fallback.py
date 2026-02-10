import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from api.binance_data import _binance_get, BINANCE_BASE_URLS

class TestBinanceFallback(unittest.TestCase):
    @patch('requests.get')
    def test_fallback_logic(self, mock_get):
        # Mock responses: 
        # 1st endpoint: 451 Protected (Geo-block)
        # 2nd endpoint: 403 Forbidden
        # 3rd endpoint: 200 OK
        
        mock_res_451 = MagicMock()
        mock_res_451.status_code = 451
        
        mock_res_403 = MagicMock()
        mock_res_403.status_code = 403
        
        mock_res_200 = MagicMock()
        mock_res_200.status_code = 200
        mock_res_200.json.return_value = {"status": "success"}
        
        mock_get.side_effect = [mock_res_451, mock_res_403, mock_res_200]
        
        result = _binance_get("test/endpoint")
        
        self.assertEqual(result, {"status": "success"})
        self.assertEqual(mock_get.call_count, 3)
        
        # Verify it tried the right URLs
        self.assertTrue(mock_get.call_args_list[0][0][0].startswith(BINANCE_BASE_URLS[0]))
        self.assertTrue(mock_get.call_args_list[1][0][0].startswith(BINANCE_BASE_URLS[1]))
        self.assertTrue(mock_get.call_args_list[2][0][0].startswith(BINANCE_BASE_URLS[2]))

    @patch('requests.get')
    def test_all_fail(self, mock_get):
        # All endpoints return 451
        mock_res_451 = MagicMock()
        mock_res_451.status_code = 451
        mock_get.return_value = mock_res_451
        
        result = _binance_get("test/endpoint")
        
        self.assertIsNone(result)
        self.assertEqual(mock_get.call_count, len(BINANCE_BASE_URLS))

if __name__ == '__main__':
    unittest.main()
