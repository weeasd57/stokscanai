#!/usr/bin/env python3
"""
Council Validator Training Script for Crypto
==============================================
Trains a meta-labeling model that filters weak KING_CRYPTO predictions.

Usage:
    python api/train_council_validator_crypto.py --symbols 20 --target-pct 0.03 --stop-pct 0.015

This script:
1. Loads the KING_CRYPTO model
2. Fetches crypto price data from Supabase
3. Applies feature engineering
4. Gets KING predictions on all data
5. Filters to KING BUY signals only
6. Labels actual outcomes (TP/SL)
7. Trains LightGBM validator
8. Saves as COUNCIL_CRYPTO.pkl
"""

import os
import sys
import warnings
import argparse
import pickle
import json
from datetime import datetime
from typing import Optional, List, Dict, Any

import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, roc_auc_score
from dotenv import load_dotenv

# Suppress warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from existing modules
from api.train_exchange_model import (
    add_technical_indicators,
    add_indicator_signals,
    add_massive_features,
    add_market_context,
    ModelTrainer,
)
from api.council_validator import make_council_validator_artifact

# Load environment variables
load_dotenv()


def load_king_model(model_path: str) -> Optional[Dict[str, Any]]:
    """Load KING_CRYPTO model from pickle file."""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ Loaded KING model from {model_path}")
        return model
    except Exception as e:
        print(f"❌ Failed to load KING model: {e}")
        return None


def get_king_predictions(king_artifact: Dict[str, Any], X: pd.DataFrame) -> np.ndarray:
    """
    Get KING model predictions (class 1 probabilities).
    
    Args:
        king_artifact: KING model artifact dictionary
        X: Feature DataFrame
        
    Returns:
        Array of probabilities for class 1 (BUY)
    """
    try:
        # Extract the actual model from artifact
        if isinstance(king_artifact, dict):
            if 'primary_model' in king_artifact:
                primary = king_artifact['primary_model']
                if isinstance(primary, dict) and 'model_str' in primary:
                    # LightGBM booster saved as string
                    import lightgbm as lgb
                    model = lgb.Booster(model_str=primary['model_str'])
                    
                    # Align features
                    feature_names = king_artifact.get('feature_names', [])
                    if feature_names:
                        missing = [f for f in feature_names if f not in X.columns]
                        for f in missing:
                            X[f] = 0
                        X = X[feature_names]
                    
                    # Predict
                    preds = model.predict(X)
                    return preds
                elif hasattr(primary, 'predict_proba'):
                    return primary.predict_proba(X)[:, 1]
            
            # Try direct model access
            if 'model' in king_artifact and hasattr(king_artifact['model'], 'predict_proba'):
                return king_artifact['model'].predict_proba(X)[:, 1]
        
        # Fallback: treat as sklearn-like model
        if hasattr(king_artifact, 'predict_proba'):
            return king_artifact.predict_proba(X)[:, 1]
        
        raise ValueError("Cannot extract predictions from KING model")
        
    except Exception as e:
        print(f"⚠️ Error getting KING predictions: {e}")
        raise


def label_outcomes(df: pd.DataFrame, target_pct: float = 0.03, stop_pct: float = 0.015, look_forward: int = 15) -> pd.Series:
    """
    Label whether a trade would hit TP before SL (Triple Barrier Method).
    
    Args:
        df: DataFrame with OHLC data
        target_pct: Take profit percentage
        stop_pct: Stop loss percentage
        look_forward: Number of periods to look ahead
        
    Returns:
        Series of binary labels (1=TP hit first, 0=SL hit first or neither)
    """
    if df.empty:
        return pd.Series(dtype=int)
    
    close_col = 'Close' if 'Close' in df.columns else 'close'
    high_col = 'High' if 'High' in df.columns else 'high'
    low_col = 'Low' if 'Low' in df.columns else 'low'
    
    close_values = pd.to_numeric(df[close_col], errors='coerce').astype(float).values
    high_values = pd.to_numeric(df[high_col], errors='coerce').astype(float).values if high_col in df.columns else close_values
    low_values = pd.to_numeric(df[low_col], errors='coerce').astype(float).values if low_col in df.columns else close_values
    
    labels = np.zeros(len(df), dtype=int)
    
    for i in range(len(df) - look_forward - 1):
        entry = float(close_values[i])
        if not np.isfinite(entry) or entry <= 0:
            continue
        
        tp = entry * (1 + float(target_pct))
        sl = entry * (1 - float(stop_pct))
        
        first_tp = None
        first_sl = None
        
        for j in range(1, look_forward + 1):
            idx = i + j
            if idx >= len(close_values):
                break
            
            hi = float(high_values[idx])
            lo = float(low_values[idx])
            
            if not np.isfinite(hi) or not np.isfinite(lo):
                continue
            
            if first_tp is None and hi >= tp:
                first_tp = j
            if first_sl is None and lo <= sl:
                first_sl = j
            
            if first_tp is not None and first_sl is not None:
                break
        
        # Label 1 if TP hit before SL
        if first_tp is not None and (first_sl is None or first_tp <= first_sl):
            labels[i] = 1
    
    return pd.Series(labels, index=df.index)


def main():
    parser = argparse.ArgumentParser(description='Train Council Validator for Crypto')
    parser.add_argument('--symbols', type=int, default=20, help='Number of crypto symbols to train on')
    parser.add_argument('--target-pct', type=float, default=0.03, help='Take profit percentage (default: 3%)')
    parser.add_argument('--stop-pct', type=float, default=0.015, help='Stop loss percentage (default: 1.5%)')
    parser.add_argument('--look-forward', type=int, default=15, help='Look forward periods for labeling')
    parser.add_argument('--king-threshold', type=float, default=0.5, help='KING prediction threshold for BUY signals')
    parser.add_argument('--approval-threshold', type=float, default=0.5, help='Validator approval threshold')
    args = parser.parse_args()
    
    print("=" * 60)
    print("🏛️ COUNCIL VALIDATOR TRAINING - CRYPTO")
    print("=" * 60)
    print(f"Target Profit: {args.target_pct*100:.1f}%")
    print(f"Stop Loss: {args.stop_pct*100:.1f}%")
    print(f"Look Forward: {args.look_forward} periods")
    print(f"KING Threshold: {args.king_threshold}")
    print("=" * 60)
    
    # 1. Load KING_CRYPTO model
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    king_path = os.path.join(models_dir, 'KING_CRYPTO 👑.pkl')
    
    if not os.path.exists(king_path):
        print(f"❌ KING_CRYPTO model not found at {king_path}")
        print("Please train KING_CRYPTO first using train_exchange_model.py")
        return
    
    king_model = load_king_model(king_path)
    if not king_model:
        return
    
    # 2. Initialize Supabase
    print("\n📡 Connecting to Supabase...")
    
    # Load from root .env file
    root_env = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    if os.path.exists(root_env):
        load_dotenv(root_env)
    
    # Get Supabase credentials
    supabase_url = os.getenv('NEXT_PUBLIC_SUPABASE_URL') or os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY') or os.getenv('NEXT_PUBLIC_SUPABASE_ANON_KEY') or os.getenv('SUPABASE_KEY')
    
    if not supabase_url or not supabase_key:
        print("❌ Supabase credentials not found in environment")
        print(f"   URL found: {supabase_url is not None}")
        print(f"   KEY found: {supabase_key is not None}")
        return
    
    print(f"✅ Supabase credentials loaded")
    
    # 3. Fetch crypto data
    print(f"\n📥 Fetching crypto data (top {args.symbols} symbols by volume)...")
    
    # Get Supabase credentials
    supabase_url = os.getenv('NEXT_PUBLIC_SUPABASE_URL') or os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY') or os.getenv('NEXT_PUBLIC_SUPABASE_ANON_KEY') or os.getenv('SUPABASE_KEY')
    
    if not supabase_url or not supabase_key:
        print("❌ Supabase credentials not found in environment")
        return
    
    trainer = ModelTrainer(
        exchange='CRYPTO',
        supabase_url=supabase_url,
        supabase_key=supabase_key
    )
    
    # Fetch intraday data (1h timeframe for crypto)
    df_all = trainer.fetch_stock_prices(use_intraday=True, timeframe='1h')
    
    if df_all.empty:
        print("❌ No data fetched from Supabase")
        return
    
    # Limit to top N symbols by volume
    symbol_volumes = df_all.groupby('symbol')['volume'].sum().sort_values(ascending=False)
    top_symbols = symbol_volumes.head(args.symbols).index.tolist()
    df_all = df_all[df_all['symbol'].isin(top_symbols)]
    
    print(f"✅ Loaded {len(df_all):,} rows for {len(top_symbols)} symbols")
    
    # 4. Load market context (BTC-USD)
    print("\n📊 Loading market context (BTC-USD)...")
    trainer.load_market_data()
    
    # 5. Process each symbol and collect meta-labeling data
    print("\n🔧 Processing symbols and generating meta-labels...")
    
    all_X = []
    all_y = []
    all_king_conf = []
    
    for symbol in top_symbols:
        try:
            df_sym = df_all[df_all['symbol'] == symbol].copy()
            
            if len(df_sym) < 200:
                continue
            
            # Ensure datetime index
            if 'date' in df_sym.columns:
                df_sym['date'] = pd.to_datetime(df_sym['date'])
                df_sym = df_sym.set_index('date').sort_index()
            
            # Feature engineering
            df_feat = add_technical_indicators(df_sym)
            if df_feat.empty:
                continue
            
            df_feat = add_indicator_signals(df_feat)
            df_feat = add_massive_features(df_feat)
            df_feat = add_market_context(df_feat, trainer.market_df)
            
            # Get KING predictions
            # Extract feature names from meta_labeling_system artifact
            feature_names = []
            if isinstance(king_model, dict):
                # Try direct feature_names
                feature_names = king_model.get('feature_names', [])
                
                # If not found, try primary_model.feature_names
                if not feature_names and 'primary_model' in king_model:
                    primary = king_model['primary_model']
                    if isinstance(primary, dict):
                        feature_names = primary.get('feature_names', [])
                
                # If still not found, try predictors
                if not feature_names:
                    feature_names = king_model.get('predictors', [])
            
            if not feature_names:
                print(f"⚠️ No feature names in KING model, skipping {symbol}")
                continue
            
            # Align features
            X_aligned = df_feat.copy()
            missing = [f for f in feature_names if f not in X_aligned.columns]
            for f in missing:
                X_aligned[f] = 0
            
            X_aligned = X_aligned[feature_names].fillna(0)
            
            # Get KING predictions
            king_probs = get_king_predictions(king_model, X_aligned)
            
            # Filter to KING BUY signals only
            buy_mask = king_probs >= args.king_threshold
            
            if buy_mask.sum() < 10:
                print(f"⚠️ Only {buy_mask.sum()} BUY signals for {symbol}, skipping")
                continue
            
            # Label actual outcomes
            outcomes = label_outcomes(df_feat, args.target_pct, args.stop_pct, args.look_forward)
            
            # Collect data for BUY signals only
            X_buy = X_aligned[buy_mask].copy()
            y_buy = outcomes[buy_mask].copy()
            king_conf_buy = king_probs[buy_mask]
            
            all_X.append(X_buy)
            all_y.append(y_buy)
            all_king_conf.append(king_conf_buy)
            
            win_rate = (y_buy == 1).sum() / len(y_buy) if len(y_buy) > 0 else 0
            print(f"  {symbol}: {buy_mask.sum()} BUY signals, {win_rate*100:.1f}% win rate")
            
        except Exception as e:
            print(f"  ❌ Error processing {symbol}: {e}")
            continue
    
    if not all_X:
        print("\n❌ No valid data collected for training")
        return
    
    # 6. Combine all data
    print("\n📦 Combining data from all symbols...")
    X_combined = pd.concat(all_X, ignore_index=True)
    y_combined = pd.concat(all_y, ignore_index=True)
    king_conf_combined = np.concatenate(all_king_conf)
    
    # Add KING confidence as a feature
    X_combined['king_confidence'] = king_conf_combined
    
    print(f"✅ Total samples: {len(X_combined):,}")
    print(f"   Positive (TP hit): {(y_combined == 1).sum():,} ({(y_combined == 1).sum() / len(y_combined) * 100:.1f}%)")
    print(f"   Negative (SL hit): {(y_combined == 0).sum():,} ({(y_combined == 0).sum() / len(y_combined) * 100:.1f}%)")
    
    # 7. Train/Test split (time-based)
    print("\n✂️ Splitting data (80/20 time-based)...")
    split_idx = int(len(X_combined) * 0.8)
    X_train = X_combined.iloc[:split_idx]
    X_test = X_combined.iloc[split_idx:]
    y_train = y_combined.iloc[:split_idx]
    y_test = y_combined.iloc[split_idx:]
    
    print(f"   Train: {len(X_train):,} samples")
    print(f"   Test:  {len(X_test):,} samples")
    
    # 8. Train LightGBM Validator
    print("\n🧠 Training Council Validator (LightGBM)...")
    
    validator_model = LGBMClassifier(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    validator_model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
    )
    
    # 9. Evaluate
    print("\n📊 Evaluation Results:")
    print("-" * 60)
    
    y_pred_train = validator_model.predict(X_train)
    y_pred_test = validator_model.predict(X_test)
    y_proba_test = validator_model.predict_proba(X_test)[:, 1]
    
    train_precision = precision_score(y_train, y_pred_train)
    train_recall = recall_score(y_train, y_pred_train)
    train_f1 = f1_score(y_train, y_pred_train)
    
    test_precision = precision_score(y_test, y_pred_test)
    test_recall = recall_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test)
    test_auc = roc_auc_score(y_test, y_proba_test)
    
    print(f"Train Metrics:")
    print(f"  Precision: {train_precision:.3f}")
    print(f"  Recall:    {train_recall:.3f}")
    print(f"  F1 Score:  {train_f1:.3f}")
    print()
    print(f"Test Metrics:")
    print(f"  Precision: {test_precision:.3f}")
    print(f"  Recall:    {test_recall:.3f}")
    print(f"  F1 Score:  {test_f1:.3f}")
    print(f"  AUC:       {test_auc:.3f}")
    print("-" * 60)
    
    # 10. Feature importance
    print("\n🔍 Top 10 Most Important Features:")
    feature_importance = pd.DataFrame({
        'feature': X_combined.columns,
        'importance': validator_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']:30s} {row['importance']:.4f}")
    
    # 11. Save model
    print("\n💾 Saving Council Validator...")
    
    artifact = make_council_validator_artifact(
        model=validator_model,
        feature_names=list(X_combined.columns),
        conf_feature='king_confidence',
        approval_threshold=args.approval_threshold,
        metadata={
            'exchange': 'CRYPTO',
            'training_date': datetime.now().isoformat(),
            'n_samples': len(X_combined),
            'n_symbols': len(top_symbols),
            'target_pct': args.target_pct,
            'stop_pct': args.stop_pct,
            'look_forward': args.look_forward,
            'king_threshold': args.king_threshold,
            'test_precision': float(test_precision),
            'test_recall': float(test_recall),
            'test_f1': float(test_f1),
            'test_auc': float(test_auc),
        }
    )
    
    output_path = os.path.join(models_dir, 'COUNCIL_CRYPTO.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(artifact, f)
    
    print(f"✅ Saved to: {output_path}")
    
    # Save model card
    model_card = {
        'name': 'COUNCIL_CRYPTO',
        'type': 'Council Validator',
        'exchange': 'CRYPTO',
        'created': datetime.now().isoformat(),
        'metrics': {
            'test_precision': float(test_precision),
            'test_recall': float(test_recall),
            'test_f1': float(test_f1),
            'test_auc': float(test_auc),
        },
        'parameters': {
            'target_pct': args.target_pct,
            'stop_pct': args.stop_pct,
            'approval_threshold': args.approval_threshold,
        }
    }
    
    card_path = output_path + '.model_card.json'
    with open(card_path, 'w') as f:
        json.dump(model_card, f, indent=2)
    
    print(f"✅ Model card saved to: {card_path}")
    
    print("\n" + "=" * 60)
    print("✅ COUNCIL VALIDATOR TRAINING COMPLETE!")
    print("=" * 60)
    print("\n📝 Next Steps:")
    print("1. Run backtest with validator:")
    print(f'   py api/backtest_radar.py --exchange CRYPTO --model "KING_CRYPTO 👑.pkl" --validator "COUNCIL_CRYPTO.pkl" --start "01/12/2025" --end "01/01/2026"')
    print("\n2. Compare results with/without validator to measure improvement")
    print("=" * 60)


if __name__ == '__main__':
    main()
