from pathlib import Path
import joblib
import pandas as pd

from .config import MODELS_DIR

MODEL_PATH = MODELS_DIR / "best_model.pkl"


class ChurnModel:
    def __init__(self, model_path: Path = MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        self.model = joblib.load(model_path)

    def predict_proba(self, features: dict) -> float:
        """
        features: single customer features as a dict
        returns: churn probability (0–1)
        """
        X = pd.DataFrame([features])
        proba = self.model.predict_proba(X)[0, 1]
        return float(proba)

    def predict_label(self, features: dict, threshold: float = 0.5) -> int:
        proba = self.predict_proba(features)
        return int(proba >= threshold)
