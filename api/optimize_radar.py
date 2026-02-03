import numpy as np
import pandas as pd
import os
import sys
import pickle
from backtest_radar import run_radar_simulation, load_model, reconstruct_meta_model
from api.stock_ai import _get_exchange_bulk_data
from api.train_exchange_model import add_technical_indicators, add_indicator_signals, add_massive_features, add_market_context, fetch_fundamentals_for_exchange
from api.stock_ai import _init_supabase, supabase
from dotenv import load_dotenv

# Load environment variables
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(base_dir, ".env"))
load_dotenv(os.path.join(base_dir, "web", ".env.local"), override=True)

def load_models_and_data(exchange="EGX", model_name="KING 👑.pkl", start_date="2024-01-01"):
    """
    Load data and models for optimization
    """
    print("🔄 Loading data and models...")
    
    # Load data
    buffer_start = "2023-01-01"
    data_map = _get_exchange_bulk_data(exchange, from_date=buffer_start)
    if not data_map:
        raise Exception("No data found")
    
    # Load fundamentals
    df_funds = pd.DataFrame()
    if supabase:
        df_funds = fetch_fundamentals_for_exchange(supabase, exchange)
    
    # Load market context for EGX
    market_df = None
    if exchange == "EGX":
        try:
            import json
            index_path = os.path.join(base_dir, "symbols_data", "EGX30-INDEX.json")
            if os.path.exists(index_path):
                with open(index_path, "r") as f:
                    idx_data = json.load(f)
                market_df = pd.DataFrame(idx_data)
                market_df['date'] = pd.to_datetime(market_df['date'])
                market_df.set_index('date', inplace=True)
        except Exception:
            pass
    
    # Load model
    models_dir = os.path.join(base_dir, "api", "models")
    model_path = os.path.join(models_dir, model_name)
    model = load_model(model_path)
    
    # Prepare combined dataframe
    all_data = []
    for symbol, df in data_map.items():
        if df.empty or len(df) < 60:
            continue
            
        try:
            original_index = df.index
            if not isinstance(original_index, pd.DatetimeIndex):
                original_index = pd.to_datetime(original_index)
            
            df_feat = add_technical_indicators(df)
            if df_feat.empty:
                continue
            
            if len(df_feat) == len(df):
                df_feat.index = original_index
            
            df_feat = add_indicator_signals(df_feat)
            df_feat = add_massive_features(df_feat)
            
            if market_df is not None:
                df_feat = add_market_context(df_feat, market_df)
            
            df_feat['symbol'] = symbol
            if not df_funds.empty:
                df_feat = df_feat.join(df_funds.set_index("symbol"), on="symbol", how="left")
            
            if len(df_feat) == len(df):
                df_feat.index = original_index
            
            df_feat = df_feat.fillna(0)
            
            # Filter to simulation period
            sim_start_dt = pd.to_datetime(start_date, dayfirst=True).tz_localize(None)
            if not isinstance(df_feat.index, pd.DatetimeIndex):
                df_feat.index = pd.to_datetime(df_feat.index, errors="coerce")
            idx_clean = pd.DatetimeIndex(df_feat.index).tz_localize(None)
            
            mask = idx_clean >= sim_start_dt
            df_sim = df_feat[mask]
            
            if not df_sim.empty:
                all_data.append(df_sim)
                
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            continue
    
    if not all_data:
        raise Exception("No valid data after processing")
    
    combined_data = pd.concat(all_data, ignore_index=False)
    
    return combined_data, model, None

def find_golden_threshold():
    print("🧪 Starting Brute-Force Optimization for KING 👑...")
    print("=" * 60)
    print(f"{'Threshold':<10} | {'Trades':<8} | {'Win Rate':<10} | {'Net Profit (EGP)':<15} | {'Note'}")
    print("-" * 60)

    # 1. تحميل الداتا والموديلات مرة واحدة (عشان السرعة)
    data, model, council_model = load_models_and_data() 
    
    best_profit = -np.inf
    best_threshold = 0.0
    
    # 2. حلقة التجربة (من 0.30 إلى 0.80)
    # بنجرب كل 5% زيادة
    for threshold in np.arange(0.30, 0.85, 0.05):
        
        # تشغيل المحاكاة بالعتبة الحالية
        stats = run_radar_simulation(
            df=data, 
            model=model, 
            council=council_model, 
            threshold=threshold
        )
        
        # حساب الربح من الصفقات المقبولة فقط
        trades_log = stats.get('Trades Log', pd.DataFrame())
        accepted_trades = trades_log[trades_log.get('Status') != 'Rejected']
        
        if not accepted_trades.empty:
            profit = accepted_trades['PnL_Pct'].sum() * 10000  # 10k per trade
            win_rate = (accepted_trades['PnL_Pct'] > 0).mean() * 100
            trades_count = len(accepted_trades)
        else:
            profit = 0
            win_rate = 0
            trades_count = 0
        
        # 3. تقييم النتيجة
        note = ""
        if profit > best_profit:
            best_profit = profit
            best_threshold = threshold
            note = "🔥 NEW HIGH!"
        elif win_rate > 60 and profit > 5000:
            note = "🛡️ SAFE ZONE"
            
        print(f"{threshold:.2f}       | {trades_count:<8} | {win_rate:.1f}%     | {profit:,.0f} EGP       | {note}")

    print("=" * 60)
    print(f"🏆 BEST SETTING: Threshold = {best_threshold:.2f} (Profit: {best_profit:,.0f} EGP)")

if __name__ == "__main__":
    find_golden_threshold()
