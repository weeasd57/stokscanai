import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from symbols_local import load_symbols_for_country
from stock_ai import (
    _get_exchange_bulk_data,
    _get_data_with_indicators_cached,
    _get_model_cached,
    _set_model_cache,
    _ensure_feature_columns,
    add_technical_indicators,
    prepare_for_ai,
    LGBM_PREDICTORS,
    RF_PREDICTORS,
)

router = APIRouter(prefix="/scan/fast", tags=["scan-fast"])

# Number of parallel workers for feature calculation
MAX_WORKERS = 12


class FastScanResult(Dict[str, Any]):
    pass


def _load_model(model_name: str):
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    model_path = model_name if os.path.isabs(model_name) else os.path.join(models_dir, model_name)

    cached = _get_model_cached(model_path)
    if cached:
        return cached

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    with open(model_path, "rb") as f:
        artifact = pickle.load(f)

    predictors: Optional[List[str]] = None
    is_lgbm = False
    model = artifact  # Default: artifact is the model itself

    # Handle dict artifact format (LightGBM booster saved as dict)
    if isinstance(artifact, dict) and artifact.get("kind") == "lgbm_booster":
        is_lgbm = True
        predictors = artifact.get("feature_names", [])
        model_str = artifact.get("model_str")
        if model_str:
            try:
                import lightgbm as lgb
                booster = lgb.Booster(model_str=model_str)
                # Wrap booster in a simple predict interface
                model = _BoosterWrapper(booster)
            except Exception as e:
                raise ValueError(f"Failed to load LightGBM booster: {e}")
        else:
            raise ValueError("Model artifact missing model_str")
    else:
        # Standard model object (sklearn-like)
        try:
            import lightgbm as lgb
            is_lgbm = isinstance(artifact, lgb.Booster) or "lightgbm" in type(artifact).__module__
        except Exception:
            is_lgbm = False

        if hasattr(artifact, "feature_names_"):
            predictors = list(getattr(artifact, "feature_names_") or [])
        if predictors is None and hasattr(artifact, "feature_name_"):
            predictors = list(getattr(artifact, "feature_name_"))
        if predictors is None and hasattr(artifact, "feature_names_in_"):
            predictors = list(getattr(artifact, "feature_names_in_"))
        if predictors is None and hasattr(artifact, "predictors"):
            try:
                predictors = list(getattr(artifact, "predictors"))
            except Exception:
                predictors = None

    # Final fallback to defaults
    if predictors is None or len(predictors) == 0:
        predictors = LGBM_PREDICTORS if is_lgbm else RF_PREDICTORS

    _set_model_cache(model_path, model, predictors, is_lgbm)
    return _get_model_cached(model_path)


class _BoosterWrapper:
    """Wrapper to give LightGBM Booster a sklearn-like predict interface."""
    def __init__(self, booster, threshold: float = 0.5):
        self.booster = booster
        self.threshold = threshold

    def predict(self, X):
        import numpy as np
        raw = self.booster.predict(X)
        return (np.asarray(raw) >= self.threshold).astype(int)

    def predict_proba(self, X):
        import numpy as np
        raw = self.booster.predict(X)
        probs = np.asarray(raw)
        # Return 2-column format: [prob_class_0, prob_class_1]
        return np.column_stack([1 - probs, probs])


def _process_symbol(
    sym: str,
    ex: str,
    name: str,
    df,
    model,
    predictors: List[str],
    min_precision: float,
) -> Optional[Dict[str, Any]]:
    """Process a single symbol - called in parallel."""
    try:
        raw = df
        if len(raw) > 500:
            raw = raw.iloc[-500:].copy()
        
        # Use only the fast add_technical_indicators (not add_massive_features)
        feat = _get_data_with_indicators_cached(sym, ex or "EGX", raw, add_technical_indicators)
        candidate = prepare_for_ai(feat)
        
        if len(candidate) < 60:
            return None
        
        candidate = candidate.iloc[[-1]].copy()
        _ensure_feature_columns(candidate, predictors)
        available_predictors = [p for p in predictors if p in candidate.columns]
        
        if not available_predictors:
            return None
        
        pred = int(model.predict(candidate[available_predictors])[0])
        prob = None
        if hasattr(model, "predict_proba"):
            try:
                prob = float(model.predict_proba(candidate[available_predictors])[0][1])
            except Exception:
                prob = None
        
        precision = prob if prob is not None else 0.5
        
        # Debug logging (first 5 symbols only to avoid spam)
        import random
        if random.random() < 0.02:  # Log ~2% of symbols
            print(f"DEBUG SCAN: {sym} | pred={pred} | prob={precision:.3f} | min_precision={min_precision}")
        
        if pred == 1 and precision >= min_precision:
            return {
                "symbol": sym,
                "exchange": ex,
                "name": name,
                "precision": precision,
                "last_close": float(candidate.iloc[-1]["Close"]),
                "signal": "BUY",
            }
        return None
    except Exception as e:
        # Log exceptions for debugging
        import random
        if random.random() < 0.05:
            print(f"DEBUG SCAN ERROR: {sym} | {e}")
        return None


@router.get("")
async def fast_scan(
    country: str = Query(default="Egypt", description="Country to scan"),
    limit: int = Query(default=200, ge=1, le=400, description="Max symbols to scan"),
    min_precision: float = Query(default=0.5, ge=0.0, le=1.0),
    model_name: str = Query(..., description="Model file name in api/models"),
    from_date: str = Query(default="2020-01-01"),
):
    start = time.time()
    try:
        symbols_data = load_symbols_for_country(country)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load symbols: {e}")

    # Warm bulk cache for all exchanges in this country
    exchanges = {str(row.get("Exchange", "")).upper() for row in symbols_data if isinstance(row, dict)}
    bulk_map: Dict[str, Dict[str, Any]] = {}
    for ex in exchanges:
        if not ex:
            continue
        bulk_map[ex] = _get_exchange_bulk_data(ex, from_date=from_date)

    model_entry = _load_model(model_name)
    if not model_entry:
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' not loaded")
    model, predictors, _ = model_entry
    if not predictors:
        raise HTTPException(status_code=400, detail="Model predictors not found")

    # Prepare symbols to process
    symbols_to_process = []
    for row in symbols_data:
        if len(symbols_to_process) >= limit:
            break
        if not isinstance(row, dict):
            continue
        sym = str(row.get("Code", row.get("Symbol", ""))).upper()
        ex = str(row.get("Exchange", ""))
        name = str(row.get("Name", sym))
        if not sym or not ex:
            continue

        df_map = bulk_map.get(ex.upper(), {})
        df = df_map.get(sym)
        if df is None or df.empty:
            continue
        
        symbols_to_process.append((sym, ex, name, df))

    scanned = len(symbols_to_process)
    results: List[FastScanResult] = []

    # Process symbols in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                _process_symbol, sym, ex, name, df, model, predictors, min_precision
            ): sym
            for sym, ex, name, df in symbols_to_process
        }
        
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    results.sort(key=lambda x: x.get("precision", 0), reverse=True)
    duration = time.time() - start
    return {
        "results": results,
        "scanned_count": scanned,
        "duration_seconds": round(duration, 2),
        "model": model_name,
        "limit": limit,
        "min_precision": min_precision,
    }
