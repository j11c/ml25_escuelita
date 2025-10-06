import numpy as np
import pandas as pd
import os
from pathlib import Path
import numpy as np

CURRENT_FILE = Path(__file__).resolve()
DATA_DIR = CURRENT_FILE / "../../../datasets/customer_purchases/"


def read_csv(filename: str):
    file = os.path.join(DATA_DIR, f"{filename}.csv")
    fullfilename = os.path.abspath(file)
    df = pd.read_csv(fullfilename)
    return df


def gen_smart_negatives(df):
    pass


def get_negatives(df):
    unique_customers = df["customer_id"].unique()
    unique_items = set(df["item_id"].unique())

    negatives = {}
    for customer in unique_customers:
        purcharsed_items = df[df["customer_id"] == customer]["item_id"].unique()
        non_purchased = unique_items - set(purcharsed_items)
        negatives[customer] = non_purchased
    return negatives


def gen_all_negatives(df):
    negatives = get_negatives(df)
    negative_lst = []
    for customer_id, item_set in negatives.items():
        negatives_for_customer = [
            {"customer_id": customer_id, "item_id": item_id, "label": 0}
            for item_id in item_set
        ]
        negative_lst.extend(negatives_for_customer)
    return pd.DataFrame(negative_lst)


def gen_random_negatives(df, n_per_positive=2):
    negatives = get_negatives(df)
    negative_lst = []
    for customer_id, item_set in negatives.items():
        rand_items = np.random.choice(list(item_set), size=n_per_positive)
        negatives_for_customer = [
            {"customer_id": customer_id, "item_id": item_id, "label": 0}
            for item_id in rand_items
        ]
        negative_lst.extend(negatives_for_customer)
    neg_df = pd.DataFrame(negative_lst)
    return neg_df


def gen_final_dataset(train_df, negatives):
    customer_columns = [
        "customer_date_of_birth",
        "customer_gender",
        "customer_signup_date",
    ]

    item_columns = [
        "item_title",
        "item_category",
        "item_price",
        "item_img_filename",
        "item_avg_rating",
        "item_num_ratings",
        "item_release_date",
    ]

    purcharse_columns = [
        "purchase_id",
        "purchase_timestamp",
        "customer_item_views",
        "purchase_item_rating",
        "purchase_device",
    ]

    # Return
    # Dataframe con labels 0 y uno y las mismas columnas que train_df
    # concatenar vertical los zeros
    # shuffle


if __name__ == "__main__":
    train_df = read_csv("customer_purchases_train")
    allnegatives = gen_all_negatives(train_df)
    print(allnegatives.info())
    randnegatives = gen_random_negatives(train_df, n_per_positive=3)
    print(randnegatives.info())
