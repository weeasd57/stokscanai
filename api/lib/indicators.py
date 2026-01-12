import pandas as pd
import numpy as np

def calculate_indicator_stats_v2(test_predictions: list):
    """
    Calculates win rates and signal counts for multiple indicators.
    Expected format of test_predictions: list of dicts with 'date', 'close', 'pred' (AI), 
    and indicators like 'rsi', 'macd', etc.
    """
    if not test_predictions:
        return {}

    df = pd.DataFrame(test_predictions)
    # Ensure we have a prediction to compare against future price
    # In this simplified version, we'll use the 'pred' field or mocked signals for rsi/macd/etc.
    
    # Let's derive some basic signals if they aren't explicitly in the data
    # or if we want to calculate them for the dashboard.
    
    results = {}
    
    # 1. RSI (Typical Buy < 30, Sell > 70)
    if 'rsi' in df.columns:
        buys = df[df['rsi'] < 30]
        sells = df[df['rsi'] > 70]
        results['rsi'] = {
            'buySignals': len(buys),
            'sellSignals': len(sells),
            'buyWinRate': _calculate_win_rate(df, buys.index)
        }
    
    # 2. MACD (Mocked or derived if MACD columns exist)
    if 'macd' in df.columns and 'macd_signal' in df.columns:
        buys = df[(df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))]
        sells = df[(df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))]
        results['macd'] = {
            'buySignals': len(buys),
            'sellSignals': len(sells),
            'buyWinRate': _calculate_win_rate(df, buys.index)
        }
    else:
        results['macd'] = {'buySignals': 0, 'sellSignals': 0, 'buyWinRate': 0}

    # 3. EMA (Price cross EMA 50)
    if 'ema50' in df.columns:
        buys = df[(df['close'] > df['ema50']) & (df['close'].shift(1) <= df['ema50'].shift(1))]
        sells = df[(df['close'] < df['ema50']) & (df['close'].shift(1) >= df['ema50'].shift(1))]
        results['ema'] = {
            'buySignals': len(buys),
            'sellSignals': len(sells),
            'buyWinRate': _calculate_win_rate(df, buys.index)
        }
    else:
        results['ema'] = {'buySignals': 0, 'sellSignals': 0, 'buyWinRate': 0}

    # 4. Bollinger Bands (Price cross Low Band)
    if 'bb_lower' in df.columns:
        buys = df[(df['close'] < df['bb_lower'])]
        sells = df[(df['close'] > df['bb_upper'])] if 'bb_upper' in df.columns else df[df['close'] < 0] # dummy
        results['bb'] = {
            'buySignals': len(buys),
            'sellSignals': len(sells),
            'buyWinRate': _calculate_win_rate(df, buys.index)
        }
    else:
        results['bb'] = {'buySignals': 0, 'sellSignals': 0, 'buyWinRate': 0}

    return results

def _calculate_win_rate(df, indices, window=5):
    """Calculates % of signals that resulted in a price increase over the next 'window' days."""
    if len(indices) == 0:
        return 0
    
    wins = 0
    valid_signals = 0
    
    for idx in indices:
        if idx + window >= len(df):
            continue
        
        entry_price = df.iloc[idx]['close']
        future_price = df.iloc[idx + window]['close']
        
        if future_price > entry_price:
            wins += 1
        valid_signals += 1
        
    return (wins / valid_signals * 100) if valid_signals > 0 else 0
