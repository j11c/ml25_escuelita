import pandas as pd
import os
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
from data_processing import read_test_data
from model import PurchaseModel
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
import numpy as np


CURRENT_FILE = Path(__file__).resolve()

RESULTS_DIR = CURRENT_FILE.parent / "test_results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

MODELS_DIR = CURRENT_FILE.parent / "trained_models"


def load(filename: str):
    """
    Load the model from MODELS_DIR/filename
    """
    filepath = Path(MODELS_DIR) / filename
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model


def run_inference(model_name: str, X):
    """
    Obtener las predicciones del modelo guardado en model_path para los datos de data_path.
    En su caso, utilicen este archivo para calcular las predicciones de data_test y subir sus resultados a la competencia de kaggle.
    """
    full_path = MODELS_DIR / model_name
    print(f"Loading model from {full_path}")
    # Cargar el modelo
    model = joblib.load(full_path)

    # Realizar la inferencia
    preds = model.predict(X)
    probs = model.predict_proba(X)

    results = pd.DataFrame(
        {"index": X.index, "prediction": preds, "probability": probs}  # índice original
    )
    return results


def plot_roc(y_true, y_proba):
    pass


if __name__ == "__main__":
    X = read_test_data()
    model_name = "xgboost_model_20251014_094039.pkl"
    model = load(model_name)
    preds = model.predict(X)

    # Guardar las preddiciones

    # random preds
    #preds = np.random.choice([0, 1], size=(len(X)))

    filename = "xgboost_predictions.csv"
    basepath = RESULTS_DIR / filename
    results = pd.DataFrame({"ID": X.index, "pred": preds})  # índice original
    results.to_csv(basepath, index=False)
    print(f"Saved predictions to {basepath}")
