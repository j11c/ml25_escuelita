import pandas as pd
import os
from pathlib import Path
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

DATA_COLLECTED_AT = datetime(2025, 9, 21).date()
CURRENT_FILE = Path(__file__).resolve()
DATA_DIR = CURRENT_FILE / "../../../datasets/customer_purchases/"

adjective_vocab = [
    "exclusive",
    "casual",
    "stylish",
    "elegant",
    "durable",
    "classic",
    "lightweight",
    "modern",
    "premium"
]


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


def df_to_numeric(df):
    data = df.copy()
    for c in data.columns:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    return data


def get_season(month): # Helper function extract_customer_features
    if month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    else:
        return 'autumn'
    

def extract_customer_features(train_df):
    adjective_vocab = [
        "exclusive",
        "casual",
        "stylish",
        "elegant",
        "durable",
        "classic",
        "lightweight",
        "modern",
        "premium"
    ]

    """
    Extrae features agregadas por cliente para usar en el modelo.
    Devuelve un DataFrame con una fila por cliente, indicando su perfil.
    """
    data = train_df.copy()
    data["customer_signup_date"] = pd.to_datetime(data["customer_signup_date"])
    data["customer_date_of_birth"] = pd.to_datetime(data["customer_date_of_birth"], errors="coerce")
    data["purchase_timestamp"] = pd.to_datetime(data["purchase_timestamp"])

    # Base: un cliente por fila
    customer_feat = data[["customer_id"]].drop_duplicates().reset_index(drop=True)

    # Edad (años)
    first_birth = data.groupby("customer_id")["customer_date_of_birth"].first()
    age = (pd.Timestamp(DATA_COLLECTED_AT) - first_birth).dt.days / 365.25
    customer_feat = customer_feat.merge(age.rename("age"), left_on="customer_id", right_index=True, how="left")

    # Antigüedad (años desde registro)
    signup_date = data.groupby("customer_id")["customer_signup_date"].first()
    seniority = (pd.Timestamp(DATA_COLLECTED_AT) - signup_date).dt.days / 365.25
    customer_feat = customer_feat.merge(seniority.rename("customer_seniority"), left_on="customer_id", right_index=True, how="left")

    # Estadísticas de precios
    stats = data.groupby("customer_id")["item_price"].agg(
        avg_purchase_cost="mean",
        var_purchase_cost="var",
        std_purchase_cost="std",
        num_purchases="count"
    )
    customer_feat = customer_feat.merge(stats, left_on="customer_id", right_index=True, how="left")

    # Porcentaje de compras por categoría
    cat_counts = (
        data.groupby(["customer_id", "item_category"]).size().unstack(fill_value=0)
    )
    cat_percent = cat_counts.div(cat_counts.sum(axis=1), axis=0)
    cat_percent.columns = [f"cat_pct_{c}" for c in cat_percent.columns]
    customer_feat = customer_feat.merge(cat_percent, left_on="customer_id", right_index=True, how="left")

    # Porcentaje de compras por color (a partir de filename tipo imgbl, imgrd...)
    data["item_color"] = data["item_img_filename"].str.extract(r"img([a-zA-Z]+)")[0]
    color_counts = (
        data.groupby(["customer_id", "item_color"]).size().unstack(fill_value=0)
    )
    color_percent = color_counts.div(color_counts.sum(axis=1), axis=0)
    color_percent.columns = [f"color_pct_{c}" for c in color_percent.columns]
    customer_feat = customer_feat.merge(color_percent, left_on="customer_id", right_index=True, how="left")

    # Obtener estacion del año por compras
    data['season'] = data['purchase_timestamp'].dt.month.apply(get_season)

    # Calcular porcentaje de compras por estación
    season_stats = (
        data.groupby(['customer_id', 'season'])
        .size()
        .unstack(fill_value=0)
    )
    # Convertir a porcentaje
    season_stats = season_stats.div(season_stats.sum(axis=1), axis=0).reset_index()
    customer_feat = customer_feat.merge(season_stats, on='customer_id', how='left')

    # Filtrar compras con fecha antes del corte (dataset tiene errores con compras futuras)
    data_time_filtered = data[data["purchase_timestamp"] <= pd.Timestamp(DATA_COLLECTED_AT)]

    # Promedio de días entre compras (solo dentro del rango válido)
    avg_days_between = (
        data_time_filtered
        .sort_values(["customer_id", "purchase_timestamp"])
        .groupby("customer_id")["purchase_timestamp"]
        .diff()
        .dt.days
        .groupby(data_time_filtered["customer_id"])
        .mean()
    )

    # Días desde la última compra hasta la fecha de corte
    last_purchase = data_time_filtered.groupby("customer_id")["purchase_timestamp"].max()
    days_since_last = (pd.Timestamp(DATA_COLLECTED_AT) - last_purchase).dt.days

    # Merge
    temporal_features = pd.DataFrame({
        "avg_days_between_purchases": avg_days_between,
        "days_since_last_purchase": days_since_last
    }).reset_index(names="customer_id")
    customer_feat = customer_feat.merge(temporal_features, on="customer_id", how="left")

    #---------------------------
    # Adjective tendencies
    #---------------------------
    # Concatenar todos los titles por cliente
    titles_by_customer = data.groupby('customer_id')['item_title'].apply(lambda x: ' '.join(x))
    # Vectorizar solo con nuestro vocabulario
    vectorizer = CountVectorizer(vocabulary=adjective_vocab)
    X_titles = vectorizer.fit_transform(titles_by_customer)
    # Convertir a DataFrame con porcentaje
    titles_df = pd.DataFrame(X_titles.toarray(), 
                            columns=vectorizer.get_feature_names_out(), 
                            index=titles_by_customer.index)
    # Pasar a porcentaje por cliente
    titles_df = titles_df.div(titles_df.sum(axis=1).replace(0, 1), axis=0).reset_index().rename(columns={'customer_id':'customer_id'})

    # Merge al perfil del cliente
    customer_feat = customer_feat.merge(titles_df, on='customer_id', how='left')

    # Guardar
    save_df(customer_feat, "customer_features.csv")
    return customer_feat


def merge_customer_profiles(train_df: pd.DataFrame, customer_feat: pd.DataFrame) -> pd.DataFrame:
    """
    Merge customer features into the main train DataFrame.
    
    Args:
        train_df: DataFrame con todas las compras, cada fila un artículo comprado por un cliente.
        customer_feat: DataFrame con el perfil del cliente, cada fila un cliente.

    Returns:
        DataFrame con las columnas del perfil agregadas a cada fila según customer_id.
    """
    # Asegurarse de que 'customer_id' sea columna y no índice
    if 'customer_id' not in train_df.columns:
        train_df = train_df.reset_index()
    if 'customer_id' not in customer_feat.columns:
        customer_feat = customer_feat.reset_index()

    # Merge
    merged_df = train_df.merge(customer_feat, on='customer_id', how='left')
    return merged_df


def process_df(df, training=True):
    """
    Investiga las siguientes funciones de SKlearn y determina si te son útiles
    - OneHotEncoder
    - StandardScaler
    - CountVectorizer
    - ColumnTransformer
    """

    adjective_vocab = [
        "exclusive",
        "casual",
        "stylish",
        "elegant",
        "durable",
        "classic",
        "lightweight",
        "modern",
        "premium"
    ]

    categorical_cols = [
        'customer_gender',
        'item_category',
        'item_img_filename'
    ]

    minmax_cols = [
        'age',
        'customer_seniority'
    ]

    standard_cols = [
        'avg_days_between_purchases',
        'days_since_last_purchase',
        'item_price',
        'avg_purchase_cost',
        'std_purchase_cost'
    ]

    # --- Create season columns ---
    df['item_release_date'] = pd.to_datetime(df['item_release_date'], errors='coerce')
    df['release_season'] = df['item_release_date'].dt.month.apply(get_season)
    season_dummies = pd.get_dummies(df['release_season'], prefix='season').astype(int)  # force 0/1
    df = pd.concat([df, season_dummies], axis=1)

    # --- Create per-row adjective flags ---
    for adj in adjective_vocab:
        adj_col_title = f"{adj}_in_title"
        df[adj_col_title] = df['item_title'].str.lower().str.contains(adj).astype(int)

    # --- Transformers ---
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_cols),
            ('minmax', MinMaxScaler(), minmax_cols),
            ('std', StandardScaler(), standard_cols),
        ],
        remainder='passthrough'  # keep other columns (percentages, seasonal, adjectives)
    )

    # ----- From scratch or reuse if training or not -------
    savepath = Path(DATA_DIR) / "preprocessor.pkl"
    if training:
        processed_array = preprocessor.fit_transform(df)
        joblib.dump(preprocessor, savepath)
    else:
        preprocessor = joblib.load(savepath) 
        processed_array = preprocessor.transform(df)

    # --- Build final DataFrame ---
    cat_features = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols).tolist()
    print(f"cat_featores: {cat_features}")
    all_features = cat_features + minmax_cols + standard_cols + [ 
        c for c in df.columns if c not in categorical_cols + minmax_cols + standard_cols
    ]

    processed_df = pd.DataFrame(processed_array, columns=all_features, index=df.index)
    return processed_df


def preprocess(raw_df, training=False): # funcion final de preprocesamiento
    """
    Agrega tu procesamiento de datos, considera si necesitas guardar valores de entrenamiento.
    Utiliza la bandera para distinguir entre preprocesamiento de entrenamiento y validación/prueba
    """
    customer_features = extract_customer_features(raw_df)
    merged_train_df = merge_customer_profiles(raw_df, customer_features)
    processed_df = process_df(merged_train_df, training)

    # select desired columns to keep and in desired order
    processed_df = processed_df[[
        'customer_gender_female', # customer profile begin
        'customer_gender_male',
        'age',
        'customer_seniority',
        'avg_days_between_purchases',
        'days_since_last_purchase',
        'avg_purchase_cost',
        'std_purchase_cost',
        'cat_pct_blouse',
        'cat_pct_dress',
        'cat_pct_jacket',
        'cat_pct_jeans',
        'cat_pct_shirt',
        'cat_pct_shoes',
        'cat_pct_skirt',
        'cat_pct_slacks',
        'cat_pct_suit',
        'cat_pct_t-shirt',
        'color_pct_b',
        'color_pct_bl',
        'color_pct_g',
        'color_pct_o',
        'color_pct_p',
        'color_pct_r',
        'color_pct_w',
        'color_pct_y',
        'autumn',
        'spring',
        'summer',
        'winter',
        'exclusive',
        'casual',
        'stylish',
        'elegant',
        'durable',
        'classic',
        'lightweight',
        'modern',
        'premium',
        'item_category_blouse', # item profile begin
        'item_category_dress',
        'item_category_jacket',
        'item_category_jeans',
        'item_category_shirt',
        'item_category_shoes',
        'item_category_skirt',
        'item_category_slacks',
        'item_category_suit',
        'item_category_t-shirt',
        'exclusive_in_title',
        'casual_in_title',
        'stylish_in_title',
        'elegant_in_title',
        'durable_in_title',
        'classic_in_title',
        'lightweight_in_title',
        'modern_in_title',
        'premium_in_title',
        'item_img_filename_imgb.jpg',
        'item_img_filename_imgbl.jpg',
        'item_img_filename_imgg.jpg',
        'item_img_filename_imgo.jpg',
        'item_img_filename_imgp.jpg',
        'item_img_filename_imgr.jpg',
        'item_img_filename_imgw.jpg',
        'item_img_filename_imgy.jpg',
        'item_price',
        'season_spring',
        'season_summer',
        'season_autumn',
        'season_winter'
        # 'label' se excluye 
    ]]

    save_df(processed_df, "processed_train.csv")
    return processed_df


def read_train_data():
    train_df = read_csv("customer_purchases_train")
    customer_feat = extract_customer_features(train_df)
    X = ...
    y = train_df["label"]
    return X, y


def read_test_data():
    test_df = read_csv("customer_purchases_test")
    customer_feat = read_csv("customer_feat.csv")
    test_df = pd.merge(test_df, customer_feat, on="customer_id")


    X_test = test_df
    return X_test


if __name__ == "__main__":
    train_df = read_csv("customer_purchases_train")
    print(train_df.info())
    test_df = read_csv("customer_purchases_test")
    print(test_df.columns)