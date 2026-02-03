import pandas as pd
import numpy as np
import sys
import os

# Set up path to include project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.council import TheCouncil

# Force UTF-8 for Windows emoji printing
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

class MockModel:
    def __init__(self, prob):
        self.prob = prob
    def predict_proba(self, X):
        return np.column_stack([1 - self.prob, self.prob])

def run_verify():
    print("Checking TheCouncil logic...")
    m1 = MockModel(np.array([0.8, 0.4]))
    m2 = MockModel(np.array([0.9, 0.2]))
    models = {"collector": m1, "king": m2}
    
    council = TheCouncil(models)
    X = pd.DataFrame({"dummy": [1, 2]})
    
    consensus = council.get_consensus(X)
    print(f"Consensus Scores: {consensus}")
    
    filtered = council.filter(X)
    print(f"Filtered Rows: {len(filtered)}")
    
    if len(filtered) == 1 and consensus[0] > 0.55:
        print("✅ Council Verification Passed!")
    else:
        print("❌ Council Verification Failed!")

if __name__ == "__main__":
    run_verify()
