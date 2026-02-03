import numpy as np
import pandas as pd
import pickle
import os
import lightgbm as lgb
from typing import Dict, List, Any, Optional

class TheCouncil:
    """
    Consensus-based noise reduction for stock opportunities.
    Implements Phase 2: Ensemble Filtering with Weighted Soft Voting.
    """
    def __init__(self, models_dict: Dict[str, Any], weights: Optional[Dict[str, float]] = None):
        """
        :param models_dict: Dictionary mapping model names (e.g., 'collector', 'king') to loaded model objects.
        :param weights: Dictionary mapping model names to their voting weights.
        """
        self.models = models_dict
        # Default weights from TODO.md
        self.weights = weights or {
            "collector": 0.25,
            "miner": 0.0,  # Placeholder/Missing
            "king": 0.40,
            "pa": 0.10     # Price Action (static for now)
        }
        self.threshold = 0.55

    @staticmethod
    def _align_X_for_artifact(X: pd.DataFrame, artifact: dict) -> pd.DataFrame:
        """
        Align X to the feature expectations of a saved artifact dict.
        Supports:
        - meta_labeling_system (uses primary_model.feature_names + categorical_features)
        - lgbm_booster (uses feature_names + categorical_features)
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        df = X.copy()

        kind = (artifact.get("kind") or "").strip().lower()
        required: List[str] = []
        categorical_features: List[str] = []

        if kind == "meta_labeling_system":
            primary_art = artifact.get("primary_model") or {}
            required = list(primary_art.get("feature_names") or [])
            categorical_features = list(primary_art.get("categorical_features") or [])
        elif kind == "lgbm_booster":
            required = list(artifact.get("feature_names") or [])
            categorical_features = list(artifact.get("categorical_features") or [])

        if not required:
            # Best-effort sanitize without breaking categoricals.
            out = df.copy()
            num_cols = out.select_dtypes(include=[np.number]).columns
            if len(num_cols) > 0:
                out[num_cols] = out[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
            return out

        cat_set = set(categorical_features or [])
        missing = [c for c in required if c not in df.columns]
        for c in missing:
            df[c] = "Unknown" if c in cat_set else 0

        df = df[required]

        cat_cols = [c for c in categorical_features if c in df.columns]
        if cat_cols:
            df[cat_cols] = (
                df[cat_cols]
                .astype(str)
                .replace(["nan", "None", ""], "Unknown")
                .fillna("Unknown")
                .astype("category")
            )

        non_cat_cols = [c for c in df.columns if c not in cat_cols]
        for col in non_cat_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if non_cat_cols:
            df[non_cat_cols] = df[non_cat_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

        return df

    @staticmethod
    def _predict_p1(model: Any, X: pd.DataFrame) -> np.ndarray:
        """
        Return class-1 probability for a model/artifact on aligned X.
        """
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            return probs[:, 1] if getattr(probs, "ndim", 1) == 2 else np.asarray(probs).reshape(-1)
        return np.asarray(model.predict(X)).astype(float)

    def get_consensus(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute Weighted Soft Voting Consensus Strength.
        ConsensusStrength = Σ(probability × weight)
        """
        if X.empty:
            return np.array([])

        total_weighted_prob = np.zeros(len(X))
        total_weight = 0.0

        for name, weight in self.weights.items():
            if weight <= 0:
                continue
            
            model = self.models.get(name)
            if model is None:
                continue

            try:
                # Saved artifacts are often dicts; never call .predict on a dict.
                if isinstance(model, dict):
                    kind = (model.get("kind") or "").strip().lower()
                    if kind == "meta_labeling_system":
                        from api.backtest_radar import reconstruct_meta_model
                        wrapper = reconstruct_meta_model(model)
                        if wrapper is None:
                            raise ValueError("Failed to reconstruct meta_labeling_system")
                        X_aligned = self._align_X_for_artifact(X, model)
                        p1 = wrapper.predict_proba(X_aligned)[:, 1]
                    elif kind == "lgbm_booster":
                        if not isinstance(model.get("model_str"), str) or not model.get("model_str"):
                            raise ValueError("Invalid lgbm_booster artifact: missing model_str")
                        booster = lgb.Booster(model_str=model.get("model_str"))
                        X_aligned = self._align_X_for_artifact(X, model)
                        p1 = np.asarray(booster.predict(X_aligned)).astype(float)
                    else:
                        # Unknown dict artifact shape (skip safely)
                        raise ValueError(f"Unsupported artifact kind: {kind or 'dict'}")
                else:
                    p1 = self._predict_p1(model, X)
                
                total_weighted_prob += p1 * weight
                total_weight += weight
            except Exception as e:
                print(f"Warning: Model {name} failed during consensus: {e}")

        if total_weight == 0:
            return np.zeros(len(X))

        # Normalize if some models are missing but we want to maintain the scale
        # However, the user defined static weights, so we divide by total_weight to get 0-1 range
        return total_weighted_prob / total_weight

    def get_detailed_votes(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Returns individual scores for each model in the council.
        """
        votes = {}
        for name, weight in self.weights.items():
            if weight <= 0: continue
            model = self.models.get(name)
            if model is None: continue
            try:
                if isinstance(model, dict):
                    kind = (model.get("kind") or "").strip().lower()
                    if kind == "meta_labeling_system":
                        from api.backtest_radar import reconstruct_meta_model
                        wrapper = reconstruct_meta_model(model)
                        if wrapper is None:
                            raise ValueError("Failed to reconstruct meta_labeling_system")
                        X_aligned = self._align_X_for_artifact(X, model)
                        votes[name] = wrapper.predict_proba(X_aligned)[:, 1]
                    elif kind == "lgbm_booster":
                        booster = lgb.Booster(model_str=model.get("model_str"))
                        X_aligned = self._align_X_for_artifact(X, model)
                        votes[name] = np.asarray(booster.predict(X_aligned)).astype(float)
                    else:
                        votes[name] = np.zeros(len(X))
                else:
                    votes[name] = self._predict_p1(model, X)
            except:
                votes[name] = np.zeros(len(X))
        return votes

    def filter(self, X: pd.DataFrame) -> pd.DataFrame:
        """Returns only rows that pass the consensus threshold."""
        scores = self.get_consensus(X)
        return X[scores >= self.threshold]
