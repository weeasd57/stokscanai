import argparse
import hashlib
import json
import os
import pickle
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from typing import List, Optional
from dotenv import load_dotenv
from pandas.api.types import is_numeric_dtype
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, precision_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

# Force UTF-8 stdout for Windows terminals to handle emojis / model names.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure `api` imports work when running from repo root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.council_validator import make_council_validator_artifact
from api.stock_ai import _LgbmBoosterClassifier, _MetaLabelingClassifier
from api.train_exchange_model import ModelTrainer

try:
    import lightgbm as lgb
except Exception:
    lgb = None


_DEFAULT_CACHE_DIR = os.path.join("api", "cache", "council_training")


def _cache_key(exchange: str, preset: str, target_pct: float, stop_loss_pct: float, look_forward_days: int) -> str:
    cfg = f"{exchange}|{preset}|{target_pct:.6f}|{stop_loss_pct:.6f}|{int(look_forward_days)}"
    return hashlib.md5(cfg.encode("utf-8")).hexdigest()


def _cache_paths(
    cache_dir: str,
    exchange: str,
    preset: str,
    target_pct: float,
    stop_loss_pct: float,
    look_forward_days: int,
) -> tuple[str, str]:
    os.makedirs(cache_dir, exist_ok=True)
    h = _cache_key(exchange, preset, target_pct, stop_loss_pct, look_forward_days)
    base = os.path.join(cache_dir, f"train_data_{h}")
    return base + ".parquet", base + ".pkl"


def _load_cached_df(parquet_path: str, pickle_path: str) -> pd.DataFrame:
    # Prefer parquet when available (fast + compact), otherwise pickle fallback.
    if os.path.exists(parquet_path):
        try:
            return pd.read_parquet(parquet_path)
        except Exception:
            pass
    if os.path.exists(pickle_path):
        try:
            return pd.read_pickle(pickle_path)
        except Exception:
            pass
    return pd.DataFrame()


def _save_cached_df(df: pd.DataFrame, parquet_path: str, pickle_path: str) -> None:
    try:
        df.to_parquet(parquet_path, index=False)
        return
    except Exception:
        pass
    try:
        df.to_pickle(pickle_path)
    except Exception:
        pass


def _clean_features_for_inference(
    X: pd.DataFrame,
    categorical_features: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Fast + safe cleaning for model inference:
    - Categorical columns: NaNs -> "Unknown", dtype -> category
    - Non-categorical columns: coerce to numeric, inf/NaN -> 0

    Important: never apply fillna(0) to categoricals (pandas will error).
    """
    df = X.copy()

    categorical_features = list(categorical_features or [])
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
    if non_cat_cols:
        # Convert in bulk (faster than per-column loops for large frames).
        df[non_cat_cols] = df[non_cat_cols].apply(pd.to_numeric, errors="coerce")
        df[non_cat_cols] = df[non_cat_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    return df


def _load_env() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(os.path.join(base_dir, ".env"))
    load_dotenv(os.path.join(base_dir, "web", ".env.local"), override=True)


def _get_supabase_creds() -> tuple[str, str]:
    url = (
        os.getenv("SUPABASE_URL")
        or os.getenv("NEXT_PUBLIC_SUPABASE_URL")
        or os.getenv("VITE_SUPABASE_URL")
        or ""
    ).strip()
    key = (
        os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        or os.getenv("SUPABASE_KEY")
        or os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")
        or os.getenv("SUPABASE_ANON_KEY")
        or ""
    ).strip()
    return url, key


def _reconstruct_classifier_from_artifact(artifact: object):
    """
    Returns (classifier, primary_feature_names, categorical_features, metadata)
    """
    meta: dict = {}

    if isinstance(artifact, dict) and artifact.get("kind") == "meta_labeling_system":
        if lgb is None:
            raise RuntimeError("lightgbm is required to reconstruct meta_labeling_system artifacts.")

        primary_art = artifact.get("primary_model") or {}
        model_str = primary_art.get("model_str")
        if not isinstance(model_str, str) or not model_str.strip():
            raise ValueError("Invalid artifact: missing primary_model.model_str")

        booster = lgb.Booster(model_str=model_str)
        primary_clf = _LgbmBoosterClassifier(booster, threshold=0.5)
        meta_model = artifact.get("meta_model")
        meta_features = artifact.get("meta_feature_names") or []
        meta_threshold = float(artifact.get("meta_threshold", 0.6))
        clf = _MetaLabelingClassifier(primary_clf, meta_model, meta_features, meta_threshold=meta_threshold)

        predictors = list(primary_art.get("feature_names") or list(booster.feature_name()) or [])
        categorical = list(primary_art.get("categorical_features") or [])

        meta = {
            "exchange": artifact.get("exchange") or primary_art.get("exchange"),
            "target_pct": artifact.get("target_pct") or primary_art.get("target_pct"),
            "stop_loss_pct": artifact.get("stop_loss_pct") or primary_art.get("stop_loss_pct"),
            "look_forward_days": artifact.get("look_forward_days") or primary_art.get("look_forward_days"),
            "feature_preset": artifact.get("feature_preset") or primary_art.get("feature_preset"),
            "base_kind": "meta_labeling_system",
        }
        return clf, predictors, categorical, meta

    if isinstance(artifact, dict) and artifact.get("kind") == "lgbm_booster":
        if lgb is None:
            raise RuntimeError("lightgbm is required to reconstruct lgbm_booster artifacts.")

        model_str = artifact.get("model_str")
        booster = lgb.Booster(model_str=model_str)
        threshold = float(artifact.get("threshold", 0.5))
        clf = _LgbmBoosterClassifier(booster, threshold=threshold)
        predictors = list(artifact.get("feature_names") or list(booster.feature_name()) or [])
        categorical = list(artifact.get("categorical_features") or [])
        meta = {
            "exchange": artifact.get("exchange"),
            "target_pct": artifact.get("target_pct"),
            "stop_loss_pct": artifact.get("stop_loss_pct"),
            "look_forward_days": artifact.get("look_forward_days"),
            "feature_preset": artifact.get("feature_preset"),
            "base_kind": "lgbm_booster",
        }
        return clf, predictors, categorical, meta

    # Raw sklearn / other model
    predictors = list(getattr(artifact, "feature_name_", []) or [])
    categorical = []
    meta = {"base_kind": str(type(artifact))}
    return artifact, predictors, categorical, meta


def main() -> int:
    _load_env()

    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-model", default=os.path.join("api", "models", "KING 👑.pkl"))
    parser.add_argument("--output", default=os.path.join("api", "models", "The_Council_Validator.pkl"))
    parser.add_argument("--exchange", default=None, help="Defaults to exchange stored in the primary model artifact.")
    parser.add_argument("--feature-preset", default=None, help="Defaults to value stored in the primary model artifact.")
    parser.add_argument("--target-pct", type=float, default=None)
    parser.add_argument("--stop-loss-pct", type=float, default=None)
    parser.add_argument("--look-forward-days", type=int, default=None)
    parser.add_argument("--buy-threshold", type=float, default=0.5, help="Base model BUY threshold for filtering rows.")
    parser.add_argument("--approval-threshold", type=float, default=0.5, help="Council approval threshold.")
    parser.add_argument("--min-trades", type=int, default=100)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--fast-mode", action="store_true", help="Use HistGradientBoosting (faster than RandomForest).")
    parser.add_argument("--cache-dir", default=_DEFAULT_CACHE_DIR, help="Disk cache dir for processed training data.")
    parser.add_argument("--no-cache", action="store_true", help="Disable disk cache.")
    parser.add_argument("--force-refresh", action="store_true", help="Ignore cache and recompute data.")
    args = parser.parse_args()

    if not os.path.exists(args.primary_model):
        print(f"❌ Primary model not found: {args.primary_model}")
        return 1

    with open(args.primary_model, "rb") as f:
        primary_artifact = pickle.load(f)

    base_clf, predictors, categorical_features, meta = _reconstruct_classifier_from_artifact(primary_artifact)

    exchange = (args.exchange or meta.get("exchange") or "").strip()
    if not exchange:
        print("❌ Could not determine exchange. Provide --exchange.")
        return 1

    feature_preset = (args.feature_preset or meta.get("feature_preset") or "extended").strip()
    target_pct = float(args.target_pct if args.target_pct is not None else (meta.get("target_pct") or 0.10))
    stop_loss_pct = float(args.stop_loss_pct if args.stop_loss_pct is not None else (meta.get("stop_loss_pct") or 0.05))
    look_forward_days = int(args.look_forward_days if args.look_forward_days is not None else (meta.get("look_forward_days") or 20))

    supabase_url, supabase_key = _get_supabase_creds()
    if not supabase_url or not supabase_key:
        print("❌ Missing Supabase credentials in env (.env / web/.env.local).")
        print("   Expected SUPABASE_URL/NEXT_PUBLIC_SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (or anon key).")
        return 1

    print("⚖️  Opening The Council Session...")
    print(f"📌 Exchange: {exchange} | Feature Preset: {feature_preset}")
    print(f"🎯 Label: target_pct={target_pct:.2%} | stop_loss_pct={stop_loss_pct:.2%} | look_forward_days={look_forward_days}")

    cache_parquet, cache_pickle = _cache_paths(
        args.cache_dir, exchange, feature_preset, target_pct, stop_loss_pct, look_forward_days
    )

    df_train = pd.DataFrame()
    if (not args.no_cache) and (not args.force_refresh):
        df_train = _load_cached_df(cache_parquet, cache_pickle)
        if not df_train.empty:
            print(f"⚡ Loaded processed data from cache: {os.path.basename(cache_parquet)}", flush=True)

    if df_train.empty:
        trainer = ModelTrainer(exchange=exchange, supabase_url=supabase_url, supabase_key=supabase_key)
        trainer.load_market_data()
        df_raw = trainer.fetch_stock_prices()
        if df_raw.empty:
            print("❌ No price data returned. Check Supabase connectivity and exchange.")
            return 1

        df_train = trainer.prepare_training_data(
            df_raw,
            target_pct=target_pct,
            stop_loss_pct=stop_loss_pct,
            look_forward_days=look_forward_days,
            preset=feature_preset,
            use_volatility_label=False,
        )
        if (not args.no_cache) and (not df_train.empty):
            _save_cached_df(df_train, cache_parquet, cache_pickle)
            print(f"💾 Saved processed data cache: {os.path.basename(cache_parquet)}", flush=True)
    if df_train.empty or "Target" not in df_train.columns:
        print("❌ Failed to prepare training data (empty or missing Target).")
        return 1

    y_true = df_train["Target"].astype(int).copy()

    # Use base model predictors if available; fall back to numeric columns.
    if not predictors:
        predictors = [c for c in df_train.columns if c not in {"Target", "symbol", "date"}]

    # Ensure all columns exist, then subset.
    for c in predictors:
        if c not in df_train.columns:
            df_train[c] = 0
    X_base = df_train[predictors].copy()

    # Clean features for inference (avoid fillna(0) on categoricals).
    X_base = _clean_features_for_inference(X_base, categorical_features=categorical_features or ["sector", "industry"])

    print("🔮 Generating Primary Predictions...")
    probs = base_clf.predict_proba(X_base)[:, 1]
    preds = (np.asarray(probs) >= float(args.buy_threshold)).astype(int)

    # Meta-features: numeric only + primary confidence (RF cannot ingest pandas categorical).
    meta_feature_names = []
    for c in predictors:
        if c in (categorical_features or []):
            continue
        try:
            if is_numeric_dtype(df_train[c]):
                meta_feature_names.append(c)
        except Exception:
            continue

    X_meta_base = df_train[meta_feature_names].copy()
    X_meta_base = X_meta_base.replace([np.inf, -np.inf], np.nan).fillna(0)

    mask = preds == 1
    X_council = X_meta_base.loc[mask].copy()
    X_council["primary_conf"] = np.asarray(probs)[mask]
    y_council = y_true.loc[mask].values

    trades = int(len(X_council))
    if trades == 0:
        print("❌ No BUY trades from base model under the chosen --buy-threshold.")
        return 1

    win_rate = float(np.sum(y_council == 1)) / float(trades)
    print(f"📉 Filtered Data: Base model suggested {trades} trades.")
    print(f"✅ Actual Winners in these trades: {int(np.sum(y_council == 1))} ({win_rate:.1%})")

    if trades < int(args.min_trades):
        print(f"⚠️ Warning: Not enough trades to train a Council. Need >= {args.min_trades}.")
        return 1

    if len(np.unique(y_council)) < 2:
        print("❌ Council training failed: only one class present for meta labels.")
        return 1

    X_train, X_test, y_train, y_test = train_test_split(
        X_council, y_council, test_size=float(args.test_size), shuffle=False
    )

    if args.fast_mode:
        print("🚀 Training The Council (HistGradientBoosting - fast mode)...")
        council_model = HistGradientBoostingClassifier(
            max_iter=200,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            early_stopping=True,
        )
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
        council_model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        print("🎓 Training The Council (Random Forest)...")
        council_model = RandomForestClassifier(
            n_estimators=150,
            max_depth=5,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        council_model.fit(X_train, y_train)

    print("\n--- 🛡️ Council Performance Report ---")
    val_preds = council_model.predict(X_test)
    precision = precision_score(y_test, val_preds, zero_division=0)
    print(f"🎯 Council Approval Precision: {precision:.1%}")
    print("تفاصيل:")
    print(classification_report(y_test, val_preds, zero_division=0))

    artifact = make_council_validator_artifact(
        model=council_model,
        feature_names=list(X_council.columns),
        conf_feature="primary_conf",
        approval_threshold=float(args.approval_threshold),
        metadata={
            "created_at": datetime.utcnow().isoformat() + "Z",
            "exchange": exchange,
            "feature_preset": feature_preset,
            "base_model_path": args.primary_model,
            "base_model_kind": meta.get("base_kind"),
            "buy_threshold": float(args.buy_threshold),
            "label": {
                "target_pct": target_pct,
                "stop_loss_pct": stop_loss_pct,
                "look_forward_days": look_forward_days,
            },
            "metrics": {"approval_precision": float(precision), "trades": trades, "win_rate": float(win_rate)},
        },
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(artifact, f)

    # Optional model card (mirrors existing style in api/models)
    try:
        card = {
            "model_name": os.path.basename(args.output),
            "created_at": artifact["metadata"].get("created_at"),
            "exchange": exchange,
            "artifact_kind": artifact.get("kind"),
            "base_model": {
                "path": args.primary_model,
                "kind": meta.get("base_kind"),
            },
            "training": artifact["metadata"].get("label"),
            "features": {
                "count": len(artifact.get("feature_names") or []),
                "conf_feature": artifact.get("conf_feature"),
            },
            "metrics": artifact["metadata"].get("metrics"),
        }
        card_path = args.output + ".model_card.json"
        with open(card_path, "w", encoding="utf-8") as cf:
            json.dump(card, cf)
    except Exception:
        pass

    print(f"\n✅ Council Member Hired! Saved as: {args.output}")
    print("Usage: If KING says Buy -> add primary_conf -> CouncilValidator -> if 1 -> Execute.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
