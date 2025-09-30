import pandas as pd
import os
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
from ml25.P01_customer_purchases.boilerplate.data_processing import read_test_data
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
import numpy as np


CURRENT_FILE = Path(__file__).resolve()

RESULTS_DIR = CURRENT_FILE.parent / "test_results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

MODELS_DIR = CURRENT_FILE.parent / "trained_models"


def load(self, filename: str):
    """
    Load the model from MODELS_DIR/filename
    """
    filepath = Path(MODELS_DIR) / filename
    model = joblib.load(filepath)
    print(f"{self.__repr__} || Model loaded from {filepath}")
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
    probs = ...

    results = pd.DataFrame(
        {"index": X.index, "prediction": preds, "probability": probs}  # índice original
    )
    return results


def plot_roc(y_true, y_proba):
    pass


if __name__ == "__main__":
    X = read_test_data()
    # model_name = "somemodelname.pkl"
    # model = load(model_name)
    # preds = model.predict(X)

    # Guardar las preddiciones
    preds = np.random.choice([0, 1], size=(len(X)))
    filename = "random_predictions.csv"
    basepath = RESULTS_DIR / filename
    results = pd.DataFrame({"ID": X.index, "pred": preds})  # índice original
    results.to_csv(basepath, index=False)
    print(f"Saved predictions to {basepath}")
