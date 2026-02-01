import argparse
import sys
import os
import pickle
import warnings
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load envs similar to main.py
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(base_dir, ".env"))
# Also try web/.env.local where Supabase keys overlap often in this project structure
load_dotenv(os.path.join(base_dir, "web", ".env.local"), override=True)

print(f"DEBUG: NEXT_PUBLIC_SUPABASE_URL found? {'NEXT_PUBLIC_SUPABASE_URL' in os.environ}")


# Add api parent dir to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.stock_ai import _get_exchange_bulk_data, _MetaLabelingClassifier
from api.train_exchange_model import add_massive_features

warnings.filterwarnings("ignore")

# Force UTF-8 stdout for Windows terminals to handle emojis
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

def load_model(model_path):
    """Loads a model from a pickle file."""
    try:
        with open(model_path, "rb") as f:
            obj = pickle.load(f)
        
        # Determine if it's a naked booster/model or a dictionary artifact
        if isinstance(obj, dict):
             # Check if it's a meta-labeling system
            if obj.get("kind") == "meta_labeling_system":
                # Reconstruct MetaLabelingClassifier
                primary_data = obj["primary_model"]
                # For this simulation, we really just need the 'predict' and 'predict_proba' behavior.
                # If the meta-model system is complex to reconstruct class-wise, we might need a wrapper.
                # But let's check `api/stock_ai.py` -> _MetaLabelingClassifier requires (primary, meta, names, threshold)
                
                # We need the actual model objects, not just the dicts if they are boosters.
                # However, the pickle usually saves the objects themselves if using standard joblib/pickle.
                # If `train_exchange_model` saved the ARTIFACT dict, the actual booster might be inside 'model_str' or passed alongside.
                # Let's verify how `stock_ai.py` loads it. 
                # It seems `stock_ai.py` expects the pickle to BE the model or the dict containing it.
                
                # Wait, `api/stock_ai.py` class `_MetaLabelingClassifier` is what we want.
                # But `save_meta_labeling_system` stores a DICT. We need to re-inflate it.
                
                return obj # We will handle dict parsing in the simulation loop or a wrapper
            return obj
        return obj
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return None

def reconstruct_meta_model(artifact):
    """
    Reconstructs a usable _MetaLabelingClassifier from the dictionary artifact.
    """
    if not isinstance(artifact, dict) or artifact.get("kind") != "meta_labeling_system":
        return None
        
    # In `train_exchange_model.py`: 
    # artifact = { "kind": ..., "primary_model": {...}, "meta_model": meta_model_obj, ... }
    # So "meta_model" IS the object (XGBClassifier or LGBMClassifier).
    
    # "primary_model" is a dict with "model_str" (if booster). 
    # We might need to inflate the primary booster from string if it was saved as string.
    # OR, if it was saved as a raw model object, we use it. 
    # The `train_exchange_model.py` saves `model_str` for the primary.
    # This implies we need to load LightGBM booster from string.
    
    import lightgbm as lgb
    
    pm_art = artifact["primary_model"]
    if pm_art.get("model_str"):
        primary_booster = lgb.Booster(model_str=pm_art["model_str"])
        # Wrap it if needed to have a similar API
        class PrimaryWrapper:
            def __init__(self, b): self.b = b
            def predict(self, X): return self.b.predict(X)
            def predict_proba(self, X): 
                raw = self.b.predict(X)
                return np.column_stack([1-raw, raw])
            
        primary_model = PrimaryWrapper(primary_booster)
    else:
        # Fallback if it wasn't a string (unlikely for "lgbm_booster" kind)
        primary_model = pm_art
        
    meta_model = artifact["meta_model"]
    meta_features = artifact["meta_feature_names"]
    threshold = artifact.get("meta_threshold", 0.6)
    
    from api.stock_ai import _MetaLabelingClassifier
    # Note: _MetaLabelingClassifier expects feature names to slice the DF.
    wrapper = _MetaLabelingClassifier(
        primary_model=primary_model,
        meta_model=meta_model,
        meta_feature_names=meta_features,
        meta_threshold=threshold
    )
    return wrapper

def run_radar_simulation(df, model, threshold=0.6, capital=100000):
    """
    Simulation of Radar: Base Model Detector -> Meta Model Confirmation.
    """
    print(f"📡 Starting Radar Simulation on {len(df)} candles...")
    
    balance = capital
    trade_log = []
    
    # Settings (from user request)
    TARGET_PCT = 0.10
    STOP_LOSS_PCT = 0.05
    HOLD_MAX_DAYS = 20
    
    # We need to simulate day by day.
    # But first, we need PREDICTIONS for all days to simulate the "Radar".
    # Since we can't easily loop row-by-row for feature gen (too slow),
    # we assume 'df' already has ALL massive features generated.
    
    # 1. Feature Prep
    # The model expects specific columns.
    # If `model` is the dictionary artifact, we need to know which columns.
    
    classifier = model
    # If it's the dict artifact, inflate it
    if isinstance(model, dict) and model.get("kind") == "meta_labeling_system":
        classifier = reconstruct_meta_model(model)
        if not classifier:
            print("Failed to reconstruct Meta Model.")
            return {}
    
    # Predict on the whole DF (or a test slice)
    # Note: In real life, we predict day-by-day. Using whole DF for 'predict' is technically look-ahead 
    # IF the features themselves leak. But `add_massive_features` uses rolling windows on PAST data, 
    # so predicting row N using row N's features is correct.
    
    # However, running `predict_proba` on 10k rows is fast.
    # Let's ensure we have the features.
    print("Pre-calculating signals...")
    try:
        # Get expected features from the model artifact or booster
        expected_features = []
        if isinstance(model, dict) and "primary_model" in model:
            expected_features = model["primary_model"].get("feature_names", [])
        
        # If we have a list of expected features, align the DF
        X = df.copy()
        
        if expected_features:
            # Check for missing features and fill with 0
            missing = set(expected_features) - set(X.columns)
            if missing:
                # Print once
                if not hasattr(run_radar_simulation, "_printed_missing"):
                    print(f"DEBUG: Missing {len(missing)} features: {sorted(list(missing))}")
                    run_radar_simulation._printed_missing = True
                for m in missing:
                    X[m] = 0
            
            # Reorder to match model's expected order
            X = X[expected_features]
            
            # Handle Categorical features specifically if the model expects them
            # We check "sector" and "industry" which are standard in this app
            for col in ["sector", "industry"]:
                if col in X.columns:
                    X[col] = X[col].astype('category')
        
        # Predict using the classifier
        probs = classifier.predict_proba(X) 
        confidences = probs[:, 1]
        
    except Exception as e:
        print(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return {}

    # Now we iterate to trade
    dates = df.index
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    symbols = df['symbol'].values if 'symbol' in df.columns else [None]*len(df)
    
    # Managing positions
    # Simple mode: 1 trade at a time? Or portfolio?
    # User said: " لو حطيت 100 ألف جنيه الشهر اللي فات" -> implies a portfolio or compounding.
    # Let's assume we take EVERY confirmed trade with a fixed position size or usage of available balance.
    # For simplicity of the "Report": We just log the individual trades and sum them up, 
    # simulating a compounding portfolio (reinvesting).
    
    start_balance = balance
    current_positions = [] # Not really needed if we just "jump" to outcome, 
                           # BUT for true PnL we should be careful about overlapping trades?
                           # The user code snippet simplifies: "For each row... see what happened".
                           # This implies "Potential Trades". If we have 100 signals, we can't take them all with 100k.
                           # Let's stick to the User's snippet logic: "update Portfolio" implies sequential processing.
    
    skipped_signals = 0
    
    for i in range(len(df) - HOLD_MAX_DAYS):
        # 1. Check Signal
        score = confidences[i]
        
        # Logic: If Meta Score >= Threshold -> CONFIRMED
        # (The "Base" signal is implicit in the MetaWrapper: it returns 0 if Base says 0)
        
        if score >= threshold:
            # ENTRY
            entry_price = closes[i]
            entry_date = dates[i]
            symbol = symbols[i]
            
            take_profit = entry_price * (1 + TARGET_PCT)
            stop_loss = entry_price * (1 - STOP_LOSS_PCT)
            
            outcome = "HOLD" # Time exit
            pnl_pct = 0.0
            exit_date = dates[i+HOLD_MAX_DAYS]
            exit_price = closes[i+HOLD_MAX_DAYS]
            
            # Look Forward
            for days_fwd in range(1, HOLD_MAX_DAYS + 1):
                idx = i + days_fwd
                if idx >= len(df): break
                
                high_f = highs[idx]
                low_f = lows[idx]
                
                # Check SL first (Conservative)
                if low_f <= stop_loss:
                    outcome = "STOP LOSS ❌"
                    pnl_pct = -STOP_LOSS_PCT
                    exit_date = dates[idx]
                    exit_price = stop_loss
                    break
                
                # Check TP
                if high_f >= take_profit:
                    outcome = "TARGET HIT 🎯"
                    pnl_pct = TARGET_PCT
                    exit_date = dates[idx]
                    exit_price = take_profit
                    break
            
            if outcome == "HOLD":
                # Exit at close of last day
                pnl_pct = (exit_price - entry_price) / entry_price
                outcome = f"TIME EXIT ({pnl_pct*100:.1f}%)"

            # Update Balance (Compounding)
            # Assumption: We put 100% of CURRENT portfolio into the trade? 
            # Risk Management usually says 10% per trade. 
            # But the user code said: `balance *= (1 + pnl)`. That means All-In per trade.
            # That's very risky but let's follow the user's "Simple Backtest" logic effectively.
            # Modification: To be realistic, if we have overlapping trades, we can't go All-In.
            # But since this is a sequence on a single DF (usually single symbol in user code loop?),
            # Wait, the user backtest function takes `df`. If `df` is ALL symbols mixed, sorting by date is crucial.
            # If `df` is one symbol, All-In is valid for that symbol's simulation.
            
            # Let's assume Fixed Allocation (e.g., 20% of capital) to allow diversification if we ran multiple symbols.
            # OR honestly, just follow the user logic: `balance *= (1 + pnl)` -> The "Growth of 1 unit".
            
            balance *= (1 + pnl_pct)
            
            trade_log.append({
                "Date": entry_date,
                "Symbol": symbol,
                "Entry": entry_price,
                "Exit": exit_price,
                "Score": round(score, 2),
                "Result": outcome,
                "PnL_Pct": pnl_pct,
                "Balance": int(balance)
            })

    # Summary
    total_return = ((balance - start_balance) / start_balance) * 100
    wins = [t for t in trade_log if t['PnL_Pct'] > 0]
    win_rate = len(wins) / len(trade_log) if trade_log else 0
    
    return {
        "Initial Capital": capital,
        "Final Balance": int(balance),
        "Total Return": f"{total_return:.2f}%",
        "Win Rate": f"{win_rate*100:.1f}%",
        "Total Trades": len(trade_log),
        "Trades Log": pd.DataFrame(trade_log)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exchange", required=True, help="Exchange code (e.g. EGX, US)")
    parser.add_argument("--model", required=True, help="Model filename in api/models/")
    parser.add_argument("--start", default="2024-01-01", help="Start date YYYY-MM-DD")
    args = parser.parse_args()
    
    # 1. Load Data
    print(f"📥 Fetching bulk data for {args.exchange} from {args.start}...")
    data_map = _get_exchange_bulk_data(args.exchange, from_date=args.start)
    if not data_map:
        print("❌ No data found.")
        return

    # Flatten for simulation? 
    # The simulation logic works best on a sorted stream of candles from ONE symbol, 
    # OR we need a complex portfolio simulator for multiple symbols.
    # To get the "Global" view, we can:
    # A) Run backtest per symbol, then aggregate.
    # B) Sort ALL candles by date and run as one stream (as if trading the market index?).
    
    # User's request: "AI Radar detected 12 confirmed opportunities last month".
    # This implies we scan ALL symbols.
    # So we should run the simulation on EVERY symbol and collect the trades.
    
    # 2. Setup Market Context & Fundamentals if needed
    from api.train_exchange_model import add_market_context, fetch_fundamentals_for_exchange
    from api.stock_ai import _init_supabase, supabase
    
    _init_supabase()
    market_df = None
    # Try to load EGX30 index for context if it's EGX
    if args.exchange == "EGX":
        try:
            # We can try to load from local json as described in model card
            index_path = os.path.join(base_dir, "symbols_data", "EGX30-INDEX.json")
            if os.path.exists(index_path):
                with open(index_path, "r") as f:
                    idx_data = json.load(f)
                market_df = pd.DataFrame(idx_data)
                market_df['date'] = pd.to_datetime(market_df['date'])
                market_df.set_index('date', inplace=True)
                print("✅ Market context (EGX30) loaded from local JSON.")
        except Exception as e:
            print(f"Warning: Could not load market context: {e}")

    # Load Fundamentals
    df_funds = pd.DataFrame()
    if supabase:
        print(f"📥 Fetching fundamentals for {args.exchange}...")
        df_funds = fetch_fundamentals_for_exchange(supabase, args.exchange)

    # 3. Load Model
    models_dir = os.path.join(base_dir, "api", "models")
    model_path = os.path.join(models_dir, args.model)
    if not os.path.exists(model_path):
        if os.path.exists(args.model):
            model_path = args.model
        else:
            print(f"❌ Model not found: {model_path}")
            return

    print(f"🧠 Loading model: {args.model}...")
    model = load_model(model_path)
    if not model:
        return

    # Use threshold from model card if possible
    sim_threshold = 0.6
    if isinstance(model, dict):
        sim_threshold = model.get("meta_threshold", 0.6)
    print(f"🎯 Using Meta Threshold: {sim_threshold}")

    # 4. Running Simulation
    from api.train_exchange_model import add_technical_indicators, add_indicator_signals
    
    all_trades = []
    count = 0
    
    for symbol, df in data_map.items():
        if df.empty or len(df) < 50: continue
        
        try:
            # Full Pipeline as in ModelTrainer.prepare_training_data
            # 1. Base Indicators
            df_feat = add_technical_indicators(df)
            if df_feat.empty: continue
            
            # 2. Advanced Signals
            df_feat = add_indicator_signals(df_feat)
            
            # 3. Massive Features (extended preset)
            df_feat = add_massive_features(df_feat)
            
            # 4. Market Context
            if market_df is not None:
                df_feat = add_market_context(df_feat, market_df)
            
            df_feat['symbol'] = symbol
            
            # 5. Fundamentals
            if not df_funds.empty:
                df_feat = pd.merge(df_feat, df_funds, on="symbol", how="left")
            
            # 6. Fill remaining NAs
            df_feat = df_feat.fillna(0)
            
            res = run_radar_simulation(df_feat, model, threshold=sim_threshold) 
            
            if res.get("Trades Log") is not None and not res["Trades Log"].empty:
                all_trades.append(res["Trades Log"])
                
        except Exception as e:
            # print(f"Error processing {symbol}: {e}")
            pass
            
        count += 1
        if count % 10 == 0:
            print(f"Processed {count}/{len(data_map)} symbols...")

    # 4. Global Report
    if not all_trades:
        print("❌ No trades found matching criteria.")
        return
        
    global_log = pd.concat(all_trades).sort_values("Date")
    
    # Recalculate Global PnL assuming Fixed Bet size (e.g. 10k per trade)
    capital_per_trade = 10000
    total_invested = len(global_log) * capital_per_trade
    
    global_log['Profit_Cash'] = capital_per_trade * global_log['PnL_Pct']
    net_profit = global_log['Profit_Cash'].sum()
    total_return_pct = (net_profit / total_invested * 100) if total_invested > 0 else 0
    
    wins = len(global_log[global_log['PnL_Pct'] > 0])
    win_rate = (wins / len(global_log)) * 100
    
    print("\n" + "="*40)
    print(" 🚀 FINAL RADAR BACKTEST REPORT ")
    print("="*40)
    print(f"Model: {args.model}")
    print(f"Exchange: {args.exchange}")
    print(f"Period: {args.start} to Present")
    print("-" * 20)
    print(f"Total Trades Detected: {len(global_log)}")
    print(f"Win Rate:              {win_rate:.1f}%")
    print(f"Avg Return per Trade:  {global_log['PnL_Pct'].mean()*100:.2f}%")
    print("-" * 20)
    print(f"Simulated Profit (Fixed 10k): {int(net_profit):,} EGP")
    print("="*40)
    
    # Show last 10 trades
    print("\nLATEST TRADES:")
    print(global_log.tail(10)[['Date', 'Symbol', 'Result', 'PnL_Pct']])
    
    # CSV dump
    out_file = f"backtest_results_{args.exchange}.csv"
    global_log.to_csv(out_file, index=False)
    print(f"\nDetailed log saved to {out_file}")

if __name__ == "__main__":
    main()
