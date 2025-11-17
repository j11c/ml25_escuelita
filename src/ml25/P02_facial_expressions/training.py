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
from dataset import get_loader
from network import Network
import pathlib

# Logging
import wandb
from datetime import datetime, timezone


file_path = pathlib.Path(__file__).parent.absolute()


def init_wandb(cfg, run_name):
    # Initialize wandb
    run = wandb.init(
        project="facial_expressions_cnn",
        config=cfg,
        name=run_name, #f"facial_expressions_cnn_{timestamp}_utc",
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
    net.eval()

    for i, batch in enumerate(val_loader, 0):
        batch_imgs = batch["transformed"]
        batch_labels = batch["label"]
        device = net.device

        batch_imgs = batch_imgs.to(device)
        batch_labels = batch_labels.to(device)
        with torch.inference_mode():
            # TODO: realiza un forward pass, calcula el loss y acumula el costo
            logits, proba = net(batch_imgs)
            loss = cost_function(logits, batch_labels)
            val_loss += loss.item()
    # TODO: Regresa el costo promedio por minibatch
    return val_loss / len(val_loader)


def train():
    now_utc = datetime.now(timezone.utc)
    timestamp = now_utc.strftime("%Y-%m-%d_%H-%M-%S-%f")
    run_name = f"facial_expressions_cnn_{timestamp}_utc"
    model_filename = f"best_model_{timestamp}.pth"

    # Hyperparametros
    cfg = {
        "training": {
            "learning_rate": 1e-4,
            "n_epochs": 100,
            "batch_size": 256,
        },
    }
    run = init_wandb(cfg, run_name)

    train_cfg = cfg.get("training")
    learning_rate = train_cfg.get("learning_rate")
    n_epochs = train_cfg.get("n_epochs")
    batch_size = train_cfg.get("batch_size")

    # Train, validation, test loaders
    train_dataset, train_loader = get_loader(
        "train", batch_size=batch_size, shuffle=True
    )
    val_dataset, val_loader = get_loader(
        "val", batch_size=batch_size, shuffle=False
    )
    print(
        f"Cargando datasets --> entrenamiento: {len(train_dataset)}, validacion: {len(val_dataset)}"
    )

    # Instanciamos tu red
    modelo = Network(input_dim=48, n_classes=7)

    # TODO: Define la funcion de costo
    criterion = nn.CrossEntropyLoss()

    # Define el optimizador
    optimizer = optim.Adam(modelo.parameters(), lr=learning_rate)

    best_epoch_loss = np.inf

    print(f"Iniciando entrenamiento... El mejor modelo se guardará como: {model_filename}")
    for epoch in range(n_epochs):
        modelo.train()
        train_loss = 0
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch: {epoch}")):
            batch_imgs = batch["transformed"]
            batch_labels = batch["label"]

            batch_imgs = batch_imgs.to(modelo.device)
            batch_labels = batch_labels.to(modelo.device)

            # TODO Zero grad, forward pass, backward pass, optimizer step
            optimizer.zero_grad()
            logits, proba = modelo(batch_imgs)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

            # TODO acumula el costo
            train_loss += loss.item()

        # TODO Calcula el costo promedio
        train_loss = train_loss / len(train_loader)
        val_loss = validation_step(val_loader, modelo, criterion)
        tqdm.write(
            f"Epoch: {epoch}, train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}"
        )

        # TODO guarda el modelo si el costo de validación es menor al mejor costo de validación
        if val_loss < best_epoch_loss:
            best_epoch_loss = val_loss
            tqdm.write(f"Nuevo mejor modelo ({best_epoch_loss:.4f}). Guardando local y en W&B...")
            modelo.save_model(model_filename)

            artifact = wandb.Artifact(
                name="fer2013-resnet18", 
                type="model",
                description=f"Mejor modelo del epoch {epoch} con loss {val_loss:.4f}"
            )
            path_to_model = file_path / "models" / model_filename
            artifact.add_file(path_to_model)
            run.log_artifact(artifact)

        run.log(
            {
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": val_loss,
            }
        )
    
    print("Entrenamiento finalizado.")
    wandb.finish()


if __name__ == "__main__":
    train()
