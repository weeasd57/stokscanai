import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class CouncilValidator:
    """
    A lightweight "validator" that approves/rejects a base model's BUY candidates.

    It is trained on rows where the base model predicted BUY, using:
    - market features (numeric-only)
    - an extra confidence feature from the base model (default: primary_conf)
    """

    model: Any
    feature_names: List[str]
    conf_feature: str = "primary_conf"
    approval_threshold: float = 0.5

    def _prepare_X(self, X: pd.DataFrame, primary_conf: Optional[np.ndarray] = None) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        df = X.copy()
        if self.conf_feature not in df.columns:
            if primary_conf is None:
                raise ValueError(f"Missing required confidence feature '{self.conf_feature}'.")
            df[self.conf_feature] = np.asarray(primary_conf)

        missing = [c for c in self.feature_names if c not in df.columns]
        for c in missing:
            df[c] = 0

        df = df[self.feature_names]
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        return df

    def predict_proba(self, X: pd.DataFrame, primary_conf: Optional[np.ndarray] = None) -> np.ndarray:
        Xp = self._prepare_X(X, primary_conf=primary_conf)
        if hasattr(self.model, "predict_proba"):
            p1 = self.model.predict_proba(Xp)[:, 1]
        else:
            p1 = np.asarray(self.model.predict(Xp)).astype(float)
        p1 = np.clip(np.asarray(p1), 0.0, 1.0)
        return np.column_stack([1 - p1, p1])

    def predict(self, X: pd.DataFrame, primary_conf: Optional[np.ndarray] = None) -> np.ndarray:
        p1 = self.predict_proba(X, primary_conf=primary_conf)[:, 1]
        return (np.asarray(p1) >= float(self.approval_threshold)).astype(int)


def make_council_validator_artifact(
    model: Any,
    feature_names: List[str],
    conf_feature: str = "primary_conf",
    approval_threshold: float = 0.5,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "kind": "council_validator",
        "model": model,
        "feature_names": list(feature_names or []),
        "conf_feature": str(conf_feature),
        "approval_threshold": float(approval_threshold),
        "metadata": dict(metadata or {}),
    }


def load_council_validator(obj: Any) -> Optional[CouncilValidator]:
    if isinstance(obj, CouncilValidator):
        return obj

    if isinstance(obj, dict) and obj.get("kind") == "council_validator":
        return CouncilValidator(
            model=obj.get("model"),
            feature_names=list(obj.get("feature_names") or []),
            conf_feature=str(obj.get("conf_feature") or "primary_conf"),
            approval_threshold=float(obj.get("approval_threshold") or 0.5),
        )

    # Allow loading a raw sklearn model (best-effort): caller must provide correct columns at inference time.
    if hasattr(obj, "predict") and hasattr(obj, "predict_proba"):
        return CouncilValidator(model=obj, feature_names=[], conf_feature="primary_conf", approval_threshold=0.5)

    return None


def load_council_validator_from_path(path: str) -> Optional[CouncilValidator]:
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return load_council_validator(obj)
    except Exception:
        return None

