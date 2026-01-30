
import os
import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
import json
import joblib
from datetime import datetime, timedelta

try:
    from api.stock_ai import _init_supabase, supabase, LGBM_PREDICTORS as PREDICTORS
except ImportError:
    # Fallback for direct execution
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from api.stock_ai import _init_supabase, supabase, LGBM_PREDICTORS as PREDICTORS


# Helper for conditional logging
def _log(msg, callback=None):
    if callback:
        try:
            callback(msg)
        except:
            print(msg)
    else:
        print(msg)

class ActiveLearner:
    def __init__(self, exchange="EGX", model_path=None, log_cb=None):
        self.exchange = exchange
        self.log_cb = log_cb
        self.model = self._load_model(model_path)
        _init_supabase()

    def _load_model(self, model_path=None):
        if model_path and os.path.exists(model_path):
            return joblib.load(model_path)
        
        # Try default model path: ../models/model_{exchange}.pkl
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
            # 1. If exchange looks like a filename, try it directly
            if self.exchange.endswith(".pkl"):
                direct_path = os.path.join(base_dir, "models", self.exchange)
                if os.path.exists(direct_path):
                    _log(f"ActiveLearner loaded specific model: {direct_path}", self.log_cb)
                    return joblib.load(direct_path)

            # 2. Default pattern
            default_path = os.path.join(base_dir, "models", f"model_{self.exchange}.pkl")
            if os.path.exists(default_path):
                _log(f"ActiveLearner loaded default model: {default_path}", self.log_cb)
                return joblib.load(default_path)
                
        except Exception as e:
            _log(f"ActiveLearner failed to load model: {e}", self.log_cb)

        return None

    def predict_with_uncertainty(self, X_row, symbol, date_str):
        """
        Predicts and returns result with uncertainty score.
        Logs the prediction to Supabase.
        """
        if not self.model:
            return None

        # Predict
        prob = self.model.predict(X_row)[0]
        prediction = 1 if prob > 0.5 else 0
        
        # Uncertainty: Closer to 0.5 means more uncertain
        # Score 0 (confident) to 1 (uncertain)
        # 0.5 -> 1.0 uncertain
        # 0.9 -> 0.2 uncertain
        uncertainty = 1.0 - (abs(prob - 0.5) * 2) 

        # Log to DB
        self._log_to_db(symbol, date_str, prediction, prob, uncertainty, X_row)

        return {
            "prediction": prediction,
            "confidence": prob,
            "uncertainty": uncertainty
        }

    def log_prediction_snapshot(self, symbol, date, prediction, confidence, features):
        """
        Logs an existing prediction (e.g. from the main pipeline) to the database.
        Calculates uncertainty automatically.
        """
        # Uncertainty: Closer to 0.5 means more uncertain
        uncertainty = 1.0 - (abs(confidence - 0.5) * 2) 
        self._log_to_db(symbol, date, prediction, confidence, uncertainty, features)

    def _log_to_db(self, symbol, date, prediction, confidence, uncertainty, features):
        if not supabase: return
        try:
            # Convert features to simple list/dict for JSON storage
            feat_list = features.tolist() if hasattr(features, "tolist") else list(features)
            
            data = {
                "date": date,
                "symbol": symbol,
                "exchange": self.exchange,
                "prediction": int(prediction),
                "confidence": float(confidence),
                "uncertainty_score": float(uncertainty),
                "features": json.dumps(feat_list),
                "model_version": "v1_adaptive" 
            }
            supabase.table("scan_results").insert(data).execute()
        except Exception as e:
            _log(f"Warning: Failed to log prediction: {e}", self.log_cb)

class ManualRetrainer:
    def __init__(self, exchange="EGX", log_cb=None):
        self.exchange = exchange
        self.log_cb = log_cb
        _init_supabase()

    def fetch_mistakes(self, lookback_days=30, model_name=None):
        """
        Fetches rows where actual_target is known and differs from prediction.
        """
        try:
            cutoff = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
            
            query = supabase.table("scan_results")\
                .select("*")\
                .eq("exchange", self.exchange)\
                .gte("created_at", cutoff)\
                .not_.is_("status", "null")\
                .not_.eq("status", "open")
            
            if model_name:
                query = query.eq("model_name", model_name)
                
            res = query.execute()
                
            data = res.data
            # Filter for incorrect predictions
            # actual_target is stored in 'actual_target' or inferred from 'status'
            # In the current update_actuals, it updates 'status' to 'win' or 'loss'
            # We need to map 'prediction' (1 or 0) to 'status' (win or loss)
            mistakes = []
            for row in data:
                pred = row.get('prediction')
                status = row.get('status')
                
                # If pred=1 (Buy) and status=loss -> mistake
                # If pred=0 (Sell/Hold) and status=win -> mistake (opportunity missed)
                # But usually we only retrain on BUY mistakes to avoid false positives
                is_mistake = False
                if pred == 1 and status == 'loss':
                    is_mistake = True
                elif pred == 0 and status == 'win':
                    is_mistake = True
                
                if is_mistake:
                    mistakes.append(row)
                    
            return mistakes
        except Exception as e:
            _log(f"Error fetching mistakes: {e}", self.log_cb)
            return []

    def retrain_on_mistakes(self, current_model, mistakes):
        if not mistakes:
            _log("No mistakes to retrain on.", self.log_cb)
            return current_model

        _log(f"Retraining on {len(mistakes)} mistakes...", self.log_cb)
        
        X = []
        y = []
        for m in mistakes:
            feats = json.loads(m['features'])
            X.append(feats)
            y.append(m['actual_target'])
            
        X = np.array(X)
        y = np.array(y)
        
        # Incremental learning
        try:
            # Clone parameters but keep training
            # Ensure n_estimators is small for the update step to avoid overfitting
            # actually for incremental, we continue training.
            # LightGBM 'refit' or 'train(init_model)' is best.
            # sklearn API .fit(init_model=...) works if supported.
            
            _log("Applying incremental update...", self.log_cb)
            # Use callback for lightgbm if possible, or just log
            current_model.fit(X, y, init_model=current_model)
            return current_model
        except Exception as e:
            _log(f"Retraining failed: {e}", self.log_cb)
            return current_model

def update_actuals(exchange="EGX", look_forward_days=1, target_pct=0.015, log_cb=None):
    """
    Checks past predictions and fills 'actual_target' using stock_prices 
    (Future Close > Current Close * (1+pct)).
    
    Default params should match your trained model strategy.
    """
    if not supabase: return
    
    try:
        # 1. Get pending predictions (actual_target IS NULL)
        # We look back up to 30 days, but ignore very recent ones (less than look_forward_days old)
        
        target_date_limit = (datetime.now() - timedelta(days=look_forward_days + 1)).strftime("%Y-%m-%d")
        
        # We need rows where actual_target is null AND date <= target_date_limit
        res = supabase.table("scan_results")\
            .select("*")\
            .eq("exchange", exchange)\
            .is_("status", "open")\
            .lte("created_at", target_date_limit)\
            .execute()
            
        pending = res.data
        if not pending:
            _log("No pending predictions to verify.", log_cb)
            return

        _log(f"Verifying {len(pending)} pending predictions...", log_cb)
        
        # Optimize: get unique symbols and fetch their price history once
        symbols = list(set(p['symbol'] for p in pending))
        
        # Fetch recent prices for these symbols
        # Note: fetching strictly needed range is complex, simpler to fetch last 45 days for these symbols
        start_fetch = (datetime.now() - timedelta(days=45)).strftime("%Y-%m-%d")
        
        # We'll fetch all prices for these symbols in one go? likely too big. Loop by symbol.
        processed_count = 0
        
        for sym in symbols:
            # Get predictions for this symbol
            sym_preds = [p for p in pending if p['symbol'] == sym]
            if not sym_preds: continue
            
            # Fetch prices
            p_res = supabase.table("stock_prices")\
                .select("date, close")\
                .eq("symbol", sym)\
                .eq("exchange", exchange)\
                .gte("date", start_fetch)\
                .execute()
                
            prices = pd.DataFrame(p_res.data)
            if prices.empty: continue
            
            prices['date'] = pd.to_datetime(prices['date'])
            prices = prices.set_index('date').sort_index()
            
            for pred in sym_preds:
                pred_date = pd.to_datetime(pred['date'])
                
                if pred_date not in prices.index:
                    continue
                    
                entry_price = prices.loc[pred_date, 'close']
                
                # Look forward
                # We need the price at T + look_forward
                # Get integer location
                try:
                    loc = prices.index.get_loc(pred_date)
                    future_loc = loc + look_forward_days
                    
                    if future_loc < len(prices):
                        future_price = prices.iloc[future_loc]['close']
                        
                        # Determine actual target
                        is_win = 1 if future_price > entry_price * (1 + target_pct) else 0
                        
                        # Update DB
                        supabase.table("scan_results")\
                            .update({"status": "win" if is_win else "loss"})\
                            .eq("id", pred['id'])\
                            .execute()
                            
                        processed_count += 1
                        # Optionally log granular progress if desired
                        # _log(f"Updated {sym} on {pred_date.date()}: {is_win}", log_cb)
                            
                except Exception as e:
                    # Date might be missing or index error
                    continue
                    
        _log(f"Verification complete. Updated {processed_count} predictions.", log_cb)

    except Exception as e:
        _log(f"Update actuals failed: {e}", log_cb)
