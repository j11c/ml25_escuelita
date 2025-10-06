# ML
import pandas as pd
from sklearn.metrics import classification_report

# Custom
from ml25.P01_customer_purchases.boilerplate.utils import setup_logger
from ml25.P01_customer_purchases.boilerplate.data_processing import read_train_data


def run_training(X, y, classifier: str):
    logger = setup_logger(f"training_{classifier}")
    logger.info("HOLA!!")
    # 1.Separar en entrenamiento y validacion

    # 2.Entrenamiento del modelo
    # model = PurchaseModel(...)

    # 5.Validacion

    # 6. Guardar modelo


if __name__ == "__main__":
    X, y = read_train_data()
    # model = ...
    # run_training(X, y, model)
