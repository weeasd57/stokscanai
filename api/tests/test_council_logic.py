import pytest
import pandas as pd
import numpy as np
# Set up path to include project root
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class MockModel:
    def __init__(self, prob):
        self.prob = prob
    def predict_proba(self, X):
        return np.column_stack([1 - self.prob, self.prob])

def test_council_voting():
    # Mock models
    m1 = MockModel(np.array([0.8, 0.4]))
    m2 = MockModel(np.array([0.9, 0.2]))
    
    models = {"collector": m1, "king": m2}
    # Weights: collector=0.25, king=0.40 (total 0.65)
    # Consensus = (p1*0.25 + p2*0.40) / 0.65
    # Row 0: (0.8*0.25 + 0.9*0.4) / 0.65 = (0.2 + 0.36) / 0.65 = 0.56 / 0.65 ≈ 0.86
    # Row 1: (0.4*0.25 + 0.2*0.4) / 0.65 = (0.1 + 0.08) / 0.65 = 0.18 / 0.65 ≈ 0.27
    
    council = TheCouncil(models)
    X = pd.DataFrame({"dummy": [1, 2]})
    
    consensus = council.get_consensus(X)
    assert len(consensus) == 2
    assert consensus[0] > 0.8
    assert consensus[1] < 0.3
    
    filtered = council.filter(X)
    assert len(filtered) == 1 # Only row 0 passes >= 0.55

def test_council_missing_model():
    m1 = MockModel(np.array([0.8]))
    models = {"collector": m1} # missing king
    
    council = TheCouncil(models)
    X = pd.DataFrame({"dummy": [1]})
    
    consensus = council.get_consensus(X)
    # Weight of collector is 0.25. Total weight with only collector is 0.25.
    # Result should be 0.8
    assert np.allclose(consensus, [0.8])
