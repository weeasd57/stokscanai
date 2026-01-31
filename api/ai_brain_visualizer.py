import os
import numpy as np
import pandas as pd
import pickle
from typing import Dict, List, Any, Optional

class AIBrainVisualizer:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model_data = None
        self.model = None
        self.predictors = []
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            print(f"Model path does not exist: {self.model_path}")
            return
        
        try:
            # Try loading with joblib first (common for scikit-learn/lightgbm)
            import joblib
            self.model_data = joblib.load(self.model_path)
        except Exception as e:
            print(f"Joblib load failed: {e}, trying pickle...")
            try:
                with open(self.model_path, "rb") as f:
                    self.model_data = pickle.load(f)
            except Exception as e2:
                print(f"Pickle load failed: {e2}")
                return

        if isinstance(self.model_data, dict) and self.model_data.get("kind") == "lgbm_booster":
            try:
                import lightgbm as lgb
                self.model = lgb.Booster(model_str=self.model_data["model_str"])
                self.predictors = self.model_data.get("feature_names", [])
            except Exception as lgb_err:
                print(f"Error initializing LGBM Booster: {lgb_err}")
        else:
            self.model = self.model_data
            # Try various ways to get features
            self.predictors = (
                getattr(self.model, "feature_names_in_", None) or 
                getattr(self.model, "feature_name_", None) or 
                getattr(self.model, "feature_names", [])
            )
            if hasattr(self.predictors, "tolist"):
                self.predictors = self.predictors.tolist()
            elif not isinstance(self.predictors, list):
                self.predictors = []
        
        print(f"Model loaded successfully. Features: {len(self.predictors)}")

    def generate_heatmap(self, feature_x: str, feature_y: str, fixed_values: Dict[str, float], grid_res: int = 50) -> Dict[str, Any]:
        if not self.model:
            return {"error": "Model failed to load"}
        if not self.predictors:
            return {"error": "No features found in model"}
        if feature_x not in self.predictors or feature_y not in self.predictors:
            return {"error": f"Invalid features: {feature_x} or {feature_y}"}

        # Define ranges (use defaults if not provided, or search training data if we had it)
        # For now, let's assume standard ranges for common indicators
        ranges = {
            "RSI": (0, 100),
            "Close": (0, 1000), # placeholder
            "Volume": (0, 1000000), # placeholder
            "SMA_50": (0, 1000),
            "Z_Score": (-4, 4),
            "BB_PctB": (0, 1),
            "feat_rsi_signal": (-1, 1)
        }
        
        range_x = ranges.get(feature_x, (-1, 1))
        range_y = ranges.get(feature_y, (-1, 1))
        
        x_vals = np.linspace(range_x[0], range_x[1], grid_res)
        y_vals = np.linspace(range_y[0], range_y[1], grid_res)
        
        xx, yy = np.meshgrid(x_vals, y_vals)
        
        # Prepare input data efficiently to avoid fragmentation warnings
        data_dict = {}
        for p in self.predictors:
            if p == feature_x:
                data_dict[p] = xx.ravel()
            elif p == feature_y:
                data_dict[p] = yy.ravel()
            else:
                data_dict[p] = np.full(grid_res * grid_res, fixed_values.get(p, 0.0))
        
        grid_data = pd.DataFrame(data_dict)
        
        # Predict using numpy values to bypass categorical mismatch in some boosters
        try:
            input_np = grid_data.values.astype(np.float32)
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(input_np)[:, 1]
            else:
                # LightGBM Booster
                probs = self.model.predict(input_np)
        except Exception as e:
            print(f"Prediction failed in heatmap: {e}")
            # Fallback to direct dataframe if numpy fails for some reason
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(grid_data)[:, 1]
            else:
                probs = self.model.predict(grid_data)
            
        return {
            "feature_x": feature_x,
            "feature_y": feature_y,
            "x_range": x_vals.tolist(),
            "y_range": y_vals.tolist(),
            "confidence_grid": probs.reshape(grid_res, grid_res).tolist(),
            "risk_zones": (probs < 0.3).reshape(grid_res, grid_res).tolist() # Simple threshold for risk
        }

    def simulate_decision(self, feature_values: Dict[str, float]) -> Dict[str, Any]:
        """Simulates a decision for a single set of feature values."""
        if not self.model:
            return {"error": "Model failed to load"}
        if not self.predictors:
            return {"error": "No features found in model"}
            
        # Build input row efficiently
        row_data = {p: [feature_values.get(p, 0.0)] for p in self.predictors}
        input_row = pd.DataFrame(row_data)
        
        try:
            # Predict
            input_np = input_row.values.astype(np.float32)
            if hasattr(self.model, "predict_proba"):
                prob = float(self.model.predict_proba(input_np)[0, 1])
            else:
                prob = float(self.model.predict(input_np)[0])
        except Exception as e:
            print(f"Prediction failed in simulation: {e}")
            # Fallback
            if hasattr(self.model, "predict_proba"):
                prob = float(self.model.predict_proba(input_row)[0, 1])
            else:
                prob = float(self.model.predict(input_row)[0])
            
        decision = "BUY" if prob > 0.7 else "WAIT" if prob > 0.4 else "AVOID"
        
        # Simple sensitivity analysis (Local Importance)
        sensitivity = {}
        delta = 0.01
        input_np_base = input_row.values.astype(np.float32)
        
        for i, p in enumerate(self.predictors):
            # Sensitivity (Local Importance) using perturbed numpy array
            perturbed_np = input_np_base.copy()
            perturbed_np[0, i] += delta
            
            try:
                if hasattr(self.model, "predict_proba"):
                    new_prob = float(self.model.predict_proba(perturbed_np)[0, 1])
                else:
                    new_prob = float(self.model.predict(perturbed_np)[0])
                sensitivity[p] = (new_prob - prob) / delta
            except Exception as se:
                # print(f"Sensitivity calc failed for {p}: {se}")
                sensitivity[p] = 0.0
            
        return {
            "confidence": prob,
            "decision": decision,
            "sensitivity": sensitivity,
            "is_risky": prob < 0.3 or (0.45 < prob < 0.55) # High uncertainty zone
        }
