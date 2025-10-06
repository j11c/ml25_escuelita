import pandas as pd
import os
from pathlib import Path
from datetime import datetime

DATA_COLLECTED_AT = datetime(2025, 9, 21).date()
CURRENT_FILE = Path(__file__).resolve()
DATA_DIR = CURRENT_FILE / "../../../datasets/customer_purchases/"


def read_csv(filename: str):
    file = os.path.join(DATA_DIR, f"{filename}.csv")
    fullfilename = os.path.abspath(file)
    df = pd.read_csv(fullfilename)
    return df


def save_df(df, filename: str):
    # Guardar
    save_path = os.path.join(DATA_DIR, filename)
    df.to_csv(save_path, index=False)
    print(f"df saved to {save_path}")


def extract_customer_features(train_df):
    # Consideren: que atributos del cliente siguen disponibles en prueba?
    save_df(customer_feat, "customer_features.csv")


def process_df(df, training=True):
    """
    Investiga las siguientes funciones de SKlearn y determina si te son útiles
    - OneHotEncoder
    - StandardScaler
    - CountVectorizer
    - ColumnTransformer
    """
    # Ejemplo de codigo para guardar y cargar archivos con pickle
    # savepath = Path(DATA_DIR) / "preprocessor.pkl"
    # if training:
    #     processed_array = preprocessor.fit_transform(df)
    #     joblib.dump(preprocessor, savepath)
    # else:
    #     preprocessor = joblib.load(savepath)
    #     processed_array = preprocessor.transform(df)

    # processed_df = pd.DataFrame(processed_array, columns=[...])
    # return processed_df


def preprocess(raw_df, training=False):
    """
    Agrega tu procesamiento de datos, considera si necesitas guardar valores de entrenamiento.
    Utiliza la bandera para distinguir entre preprocesamiento de entrenamiento y validación/prueba
    """
    processed_df = process_df(raw_df, training)
    return processed_df


def df_to_numeric(df):
    data = df.copy()
    for c in data.columns:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    return data


def read_train_data():
    train_df = read_csv("customer_purchases_train")
    customer_feat = extract_customer_features(train_df)
    ...
    return X, y


def read_test_data():
    test_df = read_csv("customer_purchases_test")
    customer_feat = read_csv("customer_feat.csv")

    # Cambiar por sus datos procesados
    # Prueba no tiene etiquetas
    X_test = test_df
    return X_test


if __name__ == "__main__":
    train_df = read_csv("customer_purchases_train")
    print(train_df.info())
    test_df = read_csv("customer_purchases_test")
    print(test_df.columns)
