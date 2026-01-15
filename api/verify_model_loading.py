import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stock_ai import train_and_predict, LGBM_PREDICTORS

def verify_loading():
    print("Verifying pre-trained model loading...")
    
    # 1. Create dummy data with 18 features + Target
    dates = pd.date_range(start="2023-01-01", periods=150)
    data = {}
    for p in LGBM_PREDICTORS:
        data[p] = np.random.randn(150)
    data["Target"] = np.random.randint(0, 2, 150)
    df = pd.DataFrame(data, index=dates)
    
    # 2. Call train_and_predict with exchange='BA'
    # This should trigger loading of model_BA.pkl
    try:
        model, predictors, test_df, preds, precision = train_and_predict(
            df, 
            exchange="BA"
        )
        
        print(f"Model type: {type(model)}")
        print(f"Predictors used: {len(predictors)}")
        print(f"Precision: {precision:.2f}")
        
        # Check if it's the expected model (LGBM)
        if "LGBMClassifier" in str(type(model)):
            print("SUCCESS: LightGBM model loaded correctly.")
        else:
            print("FAILURE: Model loaded but not of type LGBMClassifier.")
            
        if len(predictors) == 18:
            print("SUCCESS: 18 predictors used.")
        else:
            print(f"FAILURE: Expected 18 predictors, got {len(predictors)}.")
            
    except Exception as e:
        print(f"ERROR during verification: {e}")

if __name__ == "__main__":
    verify_loading()
