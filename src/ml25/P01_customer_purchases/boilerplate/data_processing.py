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

    """
    Extrae features agregadas por cliente para usar en el modelo.
    Devuelve un DataFrame con una fila por cliente.
    """
    data = train_df.copy()

    # ----------------------
    # Aseguramos tipo datetime
    # ----------------------
    data["customer_date_of_birth"] = pd.to_datetime(data["customer_date_of_birth"], errors="coerce")
    data["customer_signup_date"] = pd.to_datetime(data["customer_signup_date"], errors="coerce")

    # ----------------------
    # 1. Edad y antigüedad
    # ----------------------
    customer_feat = pd.DataFrame()
    customer_feat["customer_id"] = data["customer_id"].unique()

    # Edad
    first_birth = data.groupby("customer_id")["customer_date_of_birth"].first()
    customer_feat = customer_feat.merge(
        (DATA_COLLECTED_AT - first_birth).dt.days / 365.25,
        left_on="customer_id",
        right_index=True,
        how="left"
    )
    customer_feat.rename(columns={"customer_date_of_birth": "age"}, inplace=True)

    # Antigüedad
    first_signup = data.groupby("customer_id")["customer_signup_date"].first()
    customer_feat = customer_feat.merge(
        (DATA_COLLECTED_AT - first_signup).dt.days / 365.25,
        left_on="customer_id",
        right_index=True,
        how="left"
    )
    customer_feat.rename(columns={"customer_signup_date": "tenure"}, inplace=True)

    # ----------------------
    # 2. Estadísticas de compras
    # ----------------------
    stats = data.groupby("customer_id")["item_price"].agg(
        avg_purchase_cost="mean",
        var_purchase_cost="var",
        num_purchases="count"
    )
    customer_feat = customer_feat.merge(stats, left_on="customer_id", right_index=True, how="left")

    # ----------------------
    # 3. Porcentaje de compras por categoría
    # ----------------------
    cat_counts = data.groupby(["customer_id", "item_category"]).size().unstack(fill_value=0)
    cat_pct = cat_counts.div(cat_counts.sum(axis=1), axis=0).add_prefix("pct_cat_")
    customer_feat = customer_feat.merge(cat_pct, left_on="customer_id", right_index=True, how="left")

    # ----------------------
    # 4. One-hot del género
    # ----------------------
    gender = pd.get_dummies(data.set_index("customer_id")["customer_gender"], prefix="gender")
    gender = gender.groupby(level=0).max()  # Si hay varias filas por cliente, quedamos con 1/0
    customer_feat = customer_feat.merge(gender, left_on="customer_id", right_index=True, how="left")

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
    X = ...
    y = train_df["label"]
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
