# Data management
from pathlib import Path
import joblib
from datetime import datetime
import os

# ML


CURRENT_FILE = Path(__file__).resolve()
MODELS_DIR = CURRENT_FILE.parent / "trained_models"

MODELS_DIR.mkdir(exist_ok=True, parents=True)


class PurchaseModel:
    def __init__(self, *args, **kwargs):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def predict_proba(self, X):
        pass

    def get_config(self):
        """
        Return key hyperparameters of the model for logging.
        """
        pass

    def save(self, prefix: str):
        """
        Save the model to disk in MODELS_DIR with filename:
        <prefix>_<timestamp>.pkl

        Try to use descriptive prefix that help you keep track of the paramteters used for training to distinguish between models.
        """
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{now}.pkl"
        filepath = Path(MODELS_DIR) / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath = os.path.abspath(filepath)

        joblib.dump(self, filepath)
        print(f"{repr(self)} || Model saved to {filepath}")
        return filepath

    def load(self, filename: str):
        """
        Load the model from MODELS_DIR/filename
        """
        filepath = Path(MODELS_DIR) / filename
        model = joblib.load(filepath)
        print(f"{self.__repr__} || Model loaded from {filepath}")
        return model
