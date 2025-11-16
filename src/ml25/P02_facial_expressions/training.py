from torchvision.datasets import FER2013
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import torch.optim as optim
import torch
import torch.nn as nn
from tqdm import tqdm
from ml_clases.proyectos.P02_facial_expressions.dataset import get_loader
from ml_clases.proyectos.P02_facial_expressions.network import Network

# Logging
import wandb
from datetime import datetime, timezone


def init_wandb(cfg):
    # Initialize wandb
    now_utc = datetime.now(timezone.utc)
    timestamp = now_utc.strftime("%Y-%m-%d_%H-%M-%S-%f")

    run = wandb.init(
        project="facial_expressions_cnn",
        config=cfg,
        name=f"facial_expressions_cnn_{timestamp}_utc",
    )
    return run


def validation_step(val_loader, net, cost_function):
    """
    Realiza un epoch completo en el conjunto de validación
    args:
    - val_loader (torch.DataLoader): dataloader para los datos de validación
    - net: instancia de red neuronal de clase Network
    - cost_function (torch.nn): Función de costo a utilizar

    returns:
    - val_loss (float): el costo total (promedio por minibatch) de todos los datos de validación
    """
    val_loss = 0.0
    for i, batch in enumerate(val_loader, 0):
        batch_imgs = batch["transformed"]
        batch_labels = batch["label"]
        device = net.device
        batch_labels = batch_labels.to(device)
        with torch.inference_mode():
            # TODO: realiza un forward pass, calcula el loss y acumula el costo
            ...
    # TODO: Regresa el costo promedio por minibatch
    return ...


def train():
    # Hyperparametros
    cfg = {
        "training": {
            "learning_rate": 1e-4,
            "n_epochs": 100,
            "batch_size": 256,
        },
    }
    run = init_wandb(cfg)

    train_cfg = cfg.get("training")
    learning_rate = train_cfg.get("learning_rate")
    n_epochs = train_cfg.get("n_epochs")
    batch_size = train_cfg.get("batch_size")

    # Train, validation, test loaders
    train_dataset, train_loader = get_loader(
        "train", batch_size=batch_size, shuffle=True
    )
    val_dataset, val_loader = get_loader("val", batch_size=batch_size, shuffle=False)
    print(
        f"Cargando datasets --> entrenamiento: {len(train_dataset)}, validacion: {len(val_dataset)}"
    )

    # Instanciamos tu red
    modelo = Network(input_dim=48, n_classes=7)

    # TODO: Define la funcion de costo
    criterion = ...

    # Define el optimizador
    optimizer = ...

    best_epoch_loss = np.inf
    for epoch in range(n_epochs):
        train_loss = 0
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch: {epoch}")):
            batch_imgs = batch["transformed"]
            batch_labels = batch["label"]
            # TODO Zero grad, forward pass, backward pass, optimizer step
            ...

            # TODO acumula el costo
            ...

        # TODO Calcula el costo promedio
        train_loss = ...
        val_loss = validation_step(val_loader, modelo, criterion)
        tqdm.write(
            f"Epoch: {epoch}, train_loss: {train_loss:.2f}, val_loss: {val_loss:.2f}"
        )

        # TODO guarda el modelo si el costo de validación es menor al mejor costo de validación
        ...
        run.log(
            {
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": val_loss,
            }
        )


if __name__ == "__main__":
    train()
