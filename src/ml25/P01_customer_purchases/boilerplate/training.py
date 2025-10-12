# ML
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from time import time

# Custom
from ml25.P01_customer_purchases.boilerplate.utils import setup_logger
from ml25.P01_customer_purchases.boilerplate.data_processing import read_train_data
from ml25.P01_customer_purchases.boilerplate.model import PurchaseModel


def run_training(X, y, classifier: str = "random_forest", test_size = 0.2):
    # ---- Logger setup -----
    logger = setup_logger(f"training_{classifier}")
    logger.info("Starting training for classifier: {classifier}")
    
    # 1.Separar en entrenamiento y validacion
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)
    logger.info(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}")

    # 2.Entrenamiento del modelo
    model = PurchaseModel(model_type=classifier)
    logger.info(f"Model initialized: {model}")
    logger.info(f"Hyperparameters: {model.get_config()}")

    start_time = time()
    model.fit(X_train, y_train)
    end_time = time()
    logger.info(f"Training finished in {end_time - start_time:.2f} seconds")

    # 5.Validacion
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    logger.info(f"Validation accuracy: {acc:.4f}")
    logger.ingo("Classification report:\n" + classification_report(y_val, y_pred))

    # 6. Guardar modelo
    save_path = model.save(f"{classifier}_model")
    logger.info(f"Model saved to: {save_path}")

    return model

if __name__ == "__main__":
    X, y = read_train_data()

    classifiers = ["svm", "random_forest", "logistic", "xgboost"]
    trained_models = {}

    for classifier in classifiers:
        trained_models[classifier] = run_training(X, y, classifier=classifier)
