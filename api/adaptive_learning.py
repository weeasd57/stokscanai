import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import joblib
import lightgbm as lgb
from api.stock_ai import _init_supabase, supabase

# Global constants often used
PREDICTORS = ["open", "high", "low", "close", "volume"] # Fallback

def _log(msg, cb=None):
    print(f"DEBUG: {msg}")
    if cb:
        cb({"phase": "logging", "message": msg})

class ActiveLearner:
    def __init__(self, exchange="EGX", model_path=None, log_cb=None):
        self.exchange = exchange
        self.log_cb = log_cb
        self.model = None
        self.predictors = PREDICTORS
        self.is_booster = False
        self._load_and_setup_model(model_path)
        _init_supabase()

    def _load_and_setup_model(self, model_path=None):
        """Loads model and handles both sklearn wrapper and raw booster artifacts."""
        try:
            if not model_path:
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                model_path = os.path.join(base_dir, "models", f"model_{self.exchange}.pkl")
            
            if not os.path.exists(model_path):
                _log(f"Model not found at {model_path}", self.log_cb)
                return

            ext = os.path.splitext(model_path)[1]
            data = joblib.load(model_path)
            
            if isinstance(data, dict) and data.get("kind") == "lgbm_booster":
                # It's my new booster artifact
                self.model = lgb.Booster(model_str=data["model_str"])
                self.predictors = data.get("feature_names", PREDICTORS)
                self.is_booster = True
                _log(f"Loaded Native LGBM Booster for {self.exchange}", self.log_cb)
            else:
                self.model = data
                self.is_booster = False
                _log(f"Loaded Sklearn LGBM Model for {self.exchange}", self.log_cb)
                
        except Exception as e:
            _log(f"ActiveLearner failed to load model: {e}", self.log_cb)

    def predict_with_uncertainty(self, X_row, symbol, date_str):
        if self.model is None: return None
        
        # Format X_row as 2D array
        X = np.array(X_row).reshape(1, -1)
        
        if self.is_booster:
            prob = self.model.predict(X)[0]
        else:
            # predict_proba returns [p0, p1]
            prob = self.model.predict_proba(X)[0][1]
            
        prediction = 1 if prob > 0.5 else 0
        uncertainty = 1.0 - (abs(prob - 0.5) * 2) 

        self._log_to_db(symbol, date_str, prediction, prob, uncertainty, X_row)

        return {
            "prediction": prediction,
            "confidence": prob,
            "uncertainty": uncertainty
        }

    def _log_to_db(self, symbol, date, prediction, confidence, uncertainty, features):
        if not supabase: return
        try:
            feat_list = features.tolist() if hasattr(features, "tolist") else list(features)
            data = {
                "date": date,
                "symbol": symbol,
                "exchange": self.exchange,
                "prediction": int(prediction),
                "confidence": float(confidence),
                "uncertainty_score": float(uncertainty),
                "features": json.dumps(feat_list),
                "status": "open",
                "model_version": "v2_lgbm_native" 
            }
            supabase.table("scan_results").insert(data).execute()
        except Exception as e:
            _log(f"Warning: Failed to log prediction: {e}", self.log_cb)

class ManualRetrainer:
    def __init__(self, exchange="EGX", log_cb=None):
        self.exchange = exchange
        self.log_cb = log_cb
        _init_supabase()

    def retrain_on_mistakes(self, active_learner: ActiveLearner, mistakes):
        """
        Performs native LightGBM incremental training using the 'init_model' parameter.
        """
        if not mistakes or not active_learner.model:
            _log("No mistakes or model available for retraining.", self.log_cb)
            return None

        _log(f"Retraining on {len(mistakes)} mistakes...", self.log_cb)
        if self.log_cb:
            self.log_cb({"phase": "adaptive_learning", "message": f"Pre-processing {len(mistakes)} mistakes...", "stats": {"mistakes_count": len(mistakes)}})
        
        X = []
        y = []
        for m in mistakes:
            # Reconstruct feature vector
            feats = json.loads(m['features'])
            X.append(feats)
            y.append(m.get('actual_target', 1 if m['status'] == 'win' else 0))
            
        X = np.array(X)
        y = np.array(y)
        
        try:
            # Native LightGBM incremental update
            train_data = lgb.Dataset(X, label=y, free_raw_data=False)
            
            # Use current booster as init_model
            current_booster = active_learner.model if active_learner.is_booster else active_learner.model.booster_
            
            # Update params: very small learning rate for "fine-tuning"
            params = {
                "objective": "binary",
                "learning_rate": 0.01,
                "verbose": -1
            }
            
            # Continue training for a few iterations
            if self.log_cb:
                self.log_cb({"phase": "adaptive_learning", "message": "Running incremental LightGBM update...", "stats": {"mistakes_count": len(mistakes)}})

            new_booster = lgb.train(
                params,
                train_data,
                num_boost_round=10,
                init_model=current_booster
            )
            
            _log("Incremental update successful.", self.log_cb)
            return new_booster
        except Exception as e:
            _log(f"Retraining failed: {e}", self.log_cb)
            return None

def update_actuals(exchange="EGX", look_forward_days=20, target_pct=0.15, stop_loss_pct=0.05, log_cb=None):
    """
    Verifies past predictions with Stop-Loss awareness.
    A prediction is a 'win' only if target is hit BEFORE stop-loss is hit.
    """
    if not supabase: return
    
    try:
        # 1. Get open predictions that have enough history
        cutoff = (datetime.now() - timedelta(days=look_forward_days + 2)).strftime("%Y-%m-%d")
        res = supabase.table("scan_results")\
            .select("*")\
            .eq("exchange", exchange)\
            .eq("status", "open")\
            .lte("date", cutoff)\
            .execute()
            
        pending = res.data
        if not pending:
            _log("No pending predictions to verify.", log_cb)
            return

        _log(f"Verifying {len(pending)} predictions with SL/TP logic...", log_cb)
        
        processed_count = 0
        total_to_verify = len(pending)
        
        for i, pred in enumerate(pending):
            sym = pred['symbol']
            if log_cb:
                log_cb({
                    "phase": "adaptive_verifying", 
                    "message": f"Checking {sym} ({i+1}/{total_to_verify})",
                    "stats": {
                        "symbols_processed": i + 1,
                        "symbols_total": total_to_verify
                    }
                })
            pred_date = pd.to_datetime(pred['date'])
            
            # Fetch sufficient price history for this symbol
            end_date = (pred_date + timedelta(days=look_forward_days + 10)).strftime("%Y-%m-%d")
            p_res = supabase.table("stock_prices")\
                .select("date, open, high, low, close")\
                .eq("symbol", sym)\
                .eq("exchange", exchange)\
                .gte("date", pred['date'])\
                .lte("date", end_date)\
                .order("date", desc=False)\
                .execute()
                
            df = pd.DataFrame(p_res.data)
            if df.empty or len(df) < 2: continue
            
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            
            try:
                # Entry is Next Day Open (to match training logic)
                loc = df.index.get_loc(pred_date)
                if loc + 1 >= len(df): continue
                
                entry_price = df.iloc[loc + 1]['open']
                
                # Window for checking
                window = df.iloc[loc + 1 : loc + 1 + look_forward_days]
                if window.empty: continue
                
                # Logic: Find first day where high/low hits target or stop
                is_win = 0
                for _, row in window.iterrows():
                    # Stop loss hit?
                    if row['low'] <= entry_price * (1 - stop_loss_pct):
                        is_win = 0
                        break
                    # Target hit?
                    if row['high'] >= entry_price * (1 + target_pct):
                        is_win = 1
                        break
                
                # Update DB
                supabase.table("scan_results")\
                    .update({
                        "status": "win" if is_win else "loss",
                        "actual_target": is_win
                    })\
                    .eq("id", pred['id'])\
                    .execute()
                processed_count += 1
                
            except Exception:
                continue
                    
        _log(f"Verification complete. Updated {processed_count} predictions.", log_cb)

    except Exception as e:
        _log(f"Update actuals failed: {e}", log_cb)
