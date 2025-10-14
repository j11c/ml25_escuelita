import pandas as pd
import os
import joblib
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from negative_generation import (
    gen_final_dataset, gen_all_negatives, gen_random_negatives, gen_smart_negatives)


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
    

def extract_customer_features(df):  # extract customer features from label 1 non validation segment (X_train)
    """
    Extrae features agregadas por cliente para usar en el modelo.
    Devuelve un DataFrame con una fila por cliente, indicando su perfil.
    """

    data = df.copy()

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

    dates = [
        "customer_signup_date", 
        "customer_date_of_birth", 
        "purchase_timestamp"
    ]

    for col in dates:
        data[col] = pd.to_datetime(data[col], errors="coerce")


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

    # Renombrar columnas
    customer_feat = customer_feat.rename(columns=lambda c: c if c.startswith("customer") else f"customer_{c}")

    # Guardar
    save_df(customer_feat, "customer_features.csv")
    return customer_feat


def process_df(df, training=True):
    """
    El df ya debe tener los customer_features y del item
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
        'customer_age',
        'customer_seniority'
    ]

    standard_cols = [
        'customer_avg_days_between_purchases',
        'customer_days_since_last_purchase',
        'item_price',
        'customer_avg_purchase_cost',
        'customer_std_purchase_cost'
    ]

    # --- Create season columns ---
    df['item_release_date'] = pd.to_datetime(df['item_release_date'], errors='coerce')
    df['release_season'] = df['item_release_date'].dt.month.apply(get_season)

    season_dummies = pd.get_dummies(df['release_season'], prefix='item_season').astype(int)  # force 0/1
    df = pd.concat([df, season_dummies], axis=1)

    # --- Create per-row adjective flags ---
    for adj in adjective_vocab:
        adj_col_title = f"item_{adj}_in_title"
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
    all_features = cat_features + minmax_cols + standard_cols + [ 
        c for c in df.columns if c not in categorical_cols + minmax_cols + standard_cols
    ]

    processed_df = pd.DataFrame(processed_array, columns=all_features, index=df.index)
    return processed_df


def preprocess(raw_df, training=False): # funcion final de preprocesamiento
    if training:
        customer_features = extract_customer_features(raw_df)
        merged = pd.merge(test_df, customer_features, on="customer_id", how="left")

    processed_df = process_df(raw_df, training)

    # select desired columns to keep and in desired order
    if training:
        processed_df = processed_df[[
            'customer_id', # customer id
            'customer_gender_female', # customer profile begin
            'customer_gender_male',
            'customer_age',
            'customer_seniority',
            'customer_avg_days_between_purchases',
            'customer_days_since_last_purchase',
            'customer_avg_purchase_cost',
            'customer_std_purchase_cost',
            'customer_cat_pct_blouse',
            'customer_cat_pct_dress',
            'customer_cat_pct_jacket',
            'customer_cat_pct_jeans',
            'customer_cat_pct_shirt',
            'customer_cat_pct_shoes',
            'customer_cat_pct_skirt',
            'customer_cat_pct_slacks',
            'customer_cat_pct_suit',
            'customer_cat_pct_t-shirt',
            'customer_color_pct_b',
            'customer_color_pct_bl',
            'customer_color_pct_g',
            'customer_color_pct_o',
            'customer_color_pct_p',
            'customer_color_pct_r',
            'customer_color_pct_w',
            'customer_color_pct_y',
            'customer_autumn',
            'customer_spring',
            'customer_summer',
            'customer_winter',
            'customer_exclusive',
            'customer_casual',
            'customer_stylish',
            'customer_elegant',
            'customer_durable',
            'customer_classic',
            'customer_lightweight',
            'customer_modern',
            'customer_premium',
            'item_id', # item id
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
            'item_exclusive_in_title',
            'item_casual_in_title',
            'item_stylish_in_title',
            'item_elegant_in_title',
            'item_durable_in_title',
            'item_classic_in_title',
            'item_lightweight_in_title',
            'item_modern_in_title',
            'item_premium_in_title',
            'item_img_filename_imgb.jpg',
            'item_img_filename_imgbl.jpg',
            'item_img_filename_imgg.jpg',
            'item_img_filename_imgo.jpg',
            'item_img_filename_imgp.jpg',
            'item_img_filename_imgr.jpg',
            'item_img_filename_imgw.jpg',
            'item_img_filename_imgy.jpg',
            'item_price',
            'item_season_spring',
            'item_season_summer',
            'item_season_autumn',
            'item_season_winter',
            'label'
        ]]
        save_df(processed_df, "processed_train.csv")
    else:
        processed_df = processed_df[[
            'customer_id', # customer id
            'customer_gender_female', # customer profile begin
            'customer_gender_male',
            'customer_age',
            'customer_seniority',
            'customer_avg_days_between_purchases',
            'customer_days_since_last_purchase',
            'customer_avg_purchase_cost',
            'customer_std_purchase_cost',
            'customer_cat_pct_blouse',
            'customer_cat_pct_dress',
            'customer_cat_pct_jacket',
            'customer_cat_pct_jeans',
            'customer_cat_pct_shirt',
            'customer_cat_pct_shoes',
            'customer_cat_pct_skirt',
            'customer_cat_pct_slacks',
            'customer_cat_pct_suit',
            'customer_cat_pct_t-shirt',
            'customer_color_pct_b',
            'customer_color_pct_bl',
            'customer_color_pct_g',
            'customer_color_pct_o',
            'customer_color_pct_p',
            'customer_color_pct_r',
            'customer_color_pct_w',
            'customer_color_pct_y',
            'customer_autumn',
            'customer_spring',
            'customer_summer',
            'customer_winter',
            'customer_exclusive',
            'customer_casual',
            'customer_stylish',
            'customer_elegant',
            'customer_durable',
            'customer_classic',
            'customer_lightweight',
            'customer_modern',
            'customer_premium',
            'item_id', # item id
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
            'item_exclusive_in_title',
            'item_casual_in_title',
            'item_stylish_in_title',
            'item_elegant_in_title',
            'item_durable_in_title',
            'item_classic_in_title',
            'item_lightweight_in_title',
            'item_modern_in_title',
            'item_premium_in_title',
            'item_img_filename_imgb.jpg',
            'item_img_filename_imgbl.jpg',
            'item_img_filename_imgg.jpg',
            'item_img_filename_imgo.jpg',
            'item_img_filename_imgp.jpg',
            'item_img_filename_imgr.jpg',
            'item_img_filename_imgw.jpg',
            'item_img_filename_imgy.jpg',
            'item_price',
            'item_season_spring',
            'item_season_summer',
            'item_season_autumn',
            'item_season_winter'
            #'label'
        ]]
        save_df(processed_df, "processed_test.csv")
    
    return processed_df


def read_train_data():
    train_df = read_csv("customer_purchases_train")
    customer_feat = extract_customer_features(train_df)

    # -------------- Agregar negativos ------------------ #
    # Generar negativos
    train_df_neg = gen_smart_negatives(train_df, ratio=1.0)

    # Agregar Features del cliente
    train_df_cust = pd.merge(train_df, customer_feat, on="customer_id", how="left")

    processed_pos = preprocess(train_df_cust, training=True)
    processed_pos["label"] = 1

    # Obtener todas las columnas
    all_columns = processed_pos.columns

    # Separar los features exclusivos de los items
    item_feat = [col for col in all_columns if "item" in col]
    unique_items = processed_pos[item_feat].drop_duplicates(
        subset=[
            "item_id",
        ]
    )

    # Separar los features exclusivos de los clientes
    customer_feat = [col for col in all_columns if "customer" in col]
    unique_customers = processed_pos[customer_feat].drop_duplicates(
        subset=["customer_id"]
    )

    # Agregar los features de los items a los negativos
    processed_neg = pd.merge(
        train_df_neg,
        unique_items,
        on=["item_id"],
        how="left",
    )

    # Agregar los features de los usuarios a los negativos
    processed_neg = pd.merge(
        processed_neg,
        unique_customers,
        on=["customer_id"],
        how="left",
    )

    # Agregar etiqueta a los negativos
    processed_neg["label"] = 0

    # Combinar negativos con positivos para tener el dataset completo
    processed_full = (
        pd.concat([processed_pos, processed_neg], axis=0)
        .sample(frac=1)
        .reset_index(drop=True)
    )

    # Randomizar los datos (shuffle de las filas)
    shuffled = processed_full.sample(frac=1)
    # save_df(shuffled, "data_to_use.csv")

    # Transformar a tipo numero
    shuffled = df_to_numeric(shuffled)
    y = shuffled["label"]

    # Eliminar columnas que no sirven
    X = shuffled.drop(columns=["label", "customer_id", "item_id"])
    return X, y


def read_test_data():
    test_df = read_csv("customer_purchases_test")
    customer_feat = read_csv("customer_features")
    #test_df = pd.merge(test_df, customer_feat, on="customer_id")

    # agregar features derivados del cliente al dataset
    merged = pd.merge(test_df, customer_feat, on="customer_id", how="left")

    # Procesamiento de datos
    processed = preprocess(merged, training=False)

    # Si se requiere
    dropcols = []
    processed = processed.drop(columns=dropcols)

    return df_to_numeric(processed)


if __name__ == "__main__":
    train_df = read_csv("customer_purchases_train")
    print(train_df.info())
    test_df = read_csv("customer_purchases_test")
    print(test_df.columns)
