# Data management
from pathlib import Path
import joblib
from datetime import datetime
import os

# ML
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


CURRENT_FILE = Path(__file__).resolve()
MODELS_DIR = CURRENT_FILE.parent / "trained_models"
MODELS_DIR.mkdir(exist_ok=True, parents=True)


class PurchaseModel:
    def __init__(self, model_type="svm", **kwargs):
        """
        model_type: str in ["svm", "random_forest", "logistic", "xgboost"]
        kwargs: hyperparameters passed directly to the model
        """
        self.model_type = model_type.lower()

        if self.model_type == "svm":
            self.model = SVC(probability=True, **kwargs)
        elif self.model_type == "random_forest":
            self.model = RandomForestClassifier(**kwargs)
        elif self.model_type == "logistic":
            self.model = LogisticRegression(**kwargs)
        elif self.model_type == "xgboost":
            self.model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", **kwargs)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        self.model.predict(X)

    def predict_proba(self, X):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        else:
            raise AttributeError(f"{self.model_type} does not support predict_proba()")

    def get_config(self):
        """
        Return key hyperparameters of the model for logging.
        """
        return {
            "type": self.model_type,
            **self.model.get_params()
        }

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

    def __repr__(self):
        return f"<PurchaseModel type={self.model_type}>"