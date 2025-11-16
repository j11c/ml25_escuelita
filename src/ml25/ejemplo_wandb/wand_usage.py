import numpy as np
import pandas as pd
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import os
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from pathlib import Path
from ml25.ejemplo_wandb.dataset import HousingDataset
from ml25.ejemplo_wandb.model import Net

# conda activate mlenv
# pip install wandb
import wandb
from tqdm import tqdm

this_file_dir = Path(__file__).parent
# Modificar a donde tengan sus datos
# DATA_DIR = this_file_dir / "data"

DATA_DIR = this_file_dir / ".." / "datasets" / "house_prices"


def read_data(file):
    path = os.path.join(DATA_DIR, file)
    print(os.path.abspath(path))
    df = pd.read_csv(path)
    return df


def apply_preprocessing(dataset, feat_encoder, columns):
    """
    args:
    - dataset (pd.DataFrame): Conjunto de datos
    - feat_encoder (OrdinalEncoder): instancia de codificador para las variables de entrada ajustado con datos de entrenamiento
    returns:
    - transformed_dataset (np.array): dataset transformado
    """
    # Reemplazar valores categóricos por numéricos
    transformed_dataset = dataset.copy()
    transformed_dataset[columns] = feat_encoder.transform(dataset[columns])
    # Reemplazar NaN con -1
    transformed_dataset[np.isnan(transformed_dataset)] = -1
    return transformed_dataset.to_numpy()


def training_step(cfg, run, model, data, train_loader, criterion, optimizer):
    train_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        y_hat = model(inputs)
        loss = criterion(y_hat, labels)
        loss.backward()
        optimizer.step()

        # Sumamos el costo del minibatch para calcular el promedio
        train_loss += loss.item()
    return train_loss


def validation_step(val_loader, net, cost_function):
    """
    Realiza un epoch completo en el conjunto de validación
    args:
    - val_loader (torch.DataLoader): dataloader para los datos de validación
    - net: definición de la red neuronal (con nn.Sequential o la clase anteriormente definida)
    - cost_function(torch.nn): Función de costo a utilizar

    returns:
    - val_loss (float): el costo total (promedio por minibatch) de todos los datos de validación
    """
    val_loss = 0.0
    for i, data in enumerate(val_loader, 0):
        inputs, labels = data
        with torch.inference_mode():
            preds = net(inputs)
            loss = cost_function(preds, labels)

            # Sumamos los costos para calcular el promedio
            val_loss += loss.item()
    # Mean val loss
    return val_loss / len(val_loader)


def main(cfg):
    data = read_data("train.csv")
    full_dataset, labels = data.iloc[:, :-1], data.iloc[:, -1]

    train_data, val_data, train_labels, val_labels = train_test_split(
        full_dataset, labels, test_size=0.2, random_state=0
    )

    # Especificamos que para valores desconocidos tome -1
    # solo usamos entrenamiento apra definir el codificador
    obj_cols = train_data.dtypes == "object"
    obj_cols = list(obj_cols[obj_cols].index)
    feat_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    feat_encoder.fit(train_data[obj_cols])

    # Aplicamos el mismo preprocesamiento a todos los datasets
    train_data = apply_preprocessing(train_data, feat_encoder, obj_cols)
    val_data = apply_preprocessing(val_data, feat_encoder, obj_cols)

    # Definir dataloaders
    train_labels = train_labels.to_numpy()
    val_labels = val_labels.to_numpy()
    train_dataset = HousingDataset(train_data, train_labels)
    val_dataset = HousingDataset(val_data, val_labels)

    batch_size = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Iniciar wandb
    run = wandb.init(
        name="mi_primer_modelo",
        project="mlp_reg",
        config=cfg,
    )

    input_dims = train_dataset.input_dims
    output_dims = 1  # Regresión de un valor continuo
    model = Net(input_dims, output_dims)

    # Entrenamiento
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["training"]["lr"])

    train_loss = np.inf
    val_loss = np.inf
    epoch = 0
    pbar = tqdm(range(cfg["training"]["num_epochs"]))

    for epoch in pbar:  # loop over the dataset multiple times
        train_loss = 0.0
        # Calculamos el costo promedio
        train_loss = training_step(
            cfg, run, model, data, train_loader, criterion, optimizer
        )

        # Por cada
        val_loss = validation_step(val_loader, model, criterion)

        # Actualizamos la gráfica de las curvas de entrenamiento
        run.log(
            {
                "train_Loss": train_loss,
                "val_loss": val_loss,
                "epoch": epoch,
            }
        )
        pbar.set_description(
            f"Epoch {epoch} [Train] | Train loss {train_loss:.4f} | Val loss {val_loss:.4f}"
        )


if __name__ == "__main__":
    cfg = {
        "model": {
            "hidden_dims": 64,
            "num_layers": 3,
            "activation": "relu",
        },
        "training": {
            "batch_size": 32,
            "lr": 1e-4,
            "num_epochs": 50,
        },
    }
    main(cfg)
