# ML
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from time import time

# Custom
from utils import setup_logger
from data_processing import read_train_data
from model import PurchaseModel


MODEL_PARAMS = {
    "svm": {"C": 1.0, "kernel": "rbf", "gamma": "scale"},
    "random_forest": {"n_estimators": 200, "max_depth": 10, "random_state": 42},
    "logistic": {"C": 1.0, "solver": "lbfgs", "max_iter": 1000},
    "xgboost": {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1}
}


def run_training(X, y, classifier: str = "random_forest", test_size: float = 0.2):
    # ---- Logger setup -----
    logger = setup_logger(f"training_{classifier}")
    logger.info("Starting training for classifier: {classifier}")
    
    # 1.Separar en entrenamiento y validacion
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)
    logger.info(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}")

    # 2.Entrenamiento del modelo
    params = MODEL_PARAMS.get(classifier, {})
    model = PurchaseModel(model_type=classifier, **params)
    logger.info(f"Model initialized: {model}")
    logger.info(f"Hyperparameters: {model.get_config()}")

    start_time = time()
    model.fit(X_train, y_train)
    end_time = time()
    logger.info(f"Training finished in {end_time - start_time:.2f} seconds")

    # 5.Validacion

    # 5.1 Tarea Principal
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    logger.info(f"Validation accuracy: {acc:.4f}")
    logger.info("Classification report:\n" + classification_report(y_val, y_pred))

    # 5.2 Tarea Secundaria
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_val)[:, 1]  # Probabilidad de compra
        auc = roc_auc_score(y_val, y_proba)
        logger.info(f"Validation AUC: {auc:.4f}")
    else:
        logger.warning("Model does not support predict_proba. Cannot compute AUC.")
        auc = None

    # 6. Guardar modelo
    save_path = model.save(f"{classifier}_model")
    logger.info(f"Model saved to: {save_path}")

    return model, auc


if __name__ == "__main__":
    X, y = read_train_data()

    classifiers = [
        "svm", 
        "random_forest", 
        "logistic", 
        "xgboost"
    ]
    
    trained_models = {}
    auc_scores = {}

    for classifier in classifiers:
        trained_models[classifier], auc_scores[classifier] = run_training(X, y, classifier=classifier)
        print(f"{classifier} AUC: {auc_scores[classifier]:.4f}" if auc_scores[classifier] is not None else f"{classifier} AUC: N/A")
