import pickle
import os
import sys

# Add api to path
api_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(api_dir)

model_name = "SUPER_EGX_EXTENDED_RANDOM_3000.pkl"
models_dir = os.path.join(api_dir, "api", "models")
model_path = os.path.join(models_dir, model_name)

if not os.path.exists(model_path):
    print(f"Model not found at {model_path}")
    sys.exit(1)

with open(model_path, "rb") as f:
    artifact = pickle.load(f)

if isinstance(artifact, dict):
    print(f"Kind: {artifact.get('kind')}")
    features = artifact.get('feature_names', [])
    print(f"Number of features: {len(features)}")
    print(f"First 10 features: {features[:10]}")
    # Check for features NOT in basic set
    basic_set = {"Close", "Volume", "SMA_50", "SMA_200", "EMA_50", "EMA_200", "MACD", "MACD_Signal", "BB_Upper", "BB_Lower", "RSI", "Momentum", "VOL_SMA20", "R_VOL", "ATR_14", "ADX_14", "STOCH_K", "STOCH_D", "CCI_20", "VWAP_20", "ROC_12", "BB_PctB", "BB_Width", "OBV", "OBV_Slope", "Dist_From_High", "Dist_From_Low", "Z_Score", "Body_Size", "Upper_Shadow", "Lower_Shadow", "Day_Of_Week", "Day_Of_Month", "Close_Lag1", "Close_Diff", "RSI_Lag1", "RSI_Diff", "Volume_Lag1", "Volume_Diff", "OBV_Lag1", "OBV_Diff"}
    
    missing = [f for f in features if f not in basic_set]
    print(f"Number of features NOT in basic set: {len(missing)}")
    if missing:
        print(f"First 10 missing: {missing[:10]}")
else:
    print("Model is not a dict artifact")
