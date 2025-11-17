from torchvision.datasets import FER2013
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
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


def calculate_accuracy(logits, labels):
    _, predicted = torch.max(logits, 1)
    correct = (predicted == labels).sum().item()
    return correct


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
    correct_predictions = 0
    total_predictions = 0

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

            correct_predictions += calculate_accuracy(logits, batch_labels)
            total_predictions += batch_labels.size(0)
    
    avg_loss = val_loss / len(val_loader)
    avg_acc = (correct_predictions / total_predictions) * 100

    # TODO: Regresa el costo promedio por minibatch
    return avg_loss, avg_acc


def train(
    n_epochs: int = 100,
    learning_rate: float = 1e-5,
    batch_size: int = 256,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    base_model_path: str = None # Opcional: Para cargar un modelo previo
):
    now_utc = datetime.now(timezone.utc)
    timestamp = now_utc.strftime("%Y-%m-%d_%H-%M-%S-%f")

    run_type = "finetune" if not freeze_backbone and base_model_path else "train"
    run_name = f"fer_cnn_{run_type}_{timestamp}_utc"
    new_model_filename = f"best_model_{timestamp}.pth"

    # Hyperparametros
    cfg = {
        "training": {
            "learning_rate": 1e-5,
            "n_epochs": 150,
            "batch_size": 256,
            "pretrained": pretrained,
            "freeze_backbone": freeze_backbone,
            "base_model": base_model_path
        },
        "output_model": new_model_filename
    }
    run = init_wandb(cfg, run_name)

    print(f"--- Configuración: {run_type.upper()} ---")
    print(f"Epochs: {n_epochs} | LR: {learning_rate} | Batch: {batch_size} | Freeze: {freeze_backbone}")

    # train_cfg = cfg.get("training")
    # learning_rate = train_cfg.get("learning_rate")
    # n_epochs = train_cfg.get("n_epochs")
    # batch_size = train_cfg.get("batch_size")

    # Train, validation, test loaders
    train_dataset, train_loader = get_loader("train", batch_size=batch_size, shuffle=True)
    val_dataset, val_loader = get_loader("val", batch_size=batch_size, shuffle=False)
    print(f"Cargando datasets --> entrenamiento: {len(train_dataset)}, validacion: {len(val_dataset)}")

    # Instanciamos tu red
    modelo = Network(
        input_dim=48, 
        n_classes=7,
        pretrained=pretrained,
        freeze=freeze_backbone
    )

    # 4. Cargar pesos previos (Si aplica)
    if base_model_path:
        full_path = file_path / "models" / base_model_path
        if full_path.exists():
            print(f"Cargando pesos base desde: {base_model_path}")
            state_dict = torch.load(full_path, map_location=modelo.device)
            modelo.load_state_dict(state_dict)
        else:
            print(f"ALERTA: No se encontró modelo base: {base_model_path}. Iniciando desde cero.")


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(modelo.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = np.inf
    best_val_acc = 0.0

    print(f"Iniciando entrenamiento... El mejor modelo se guardará como: {new_model_filename}")
    for epoch in range(n_epochs):
        modelo.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}")

        for batch in loop: #enumerate(tqdm(train_loader, desc=f"Epoch: {epoch}")):
            batch_imgs = batch["transformed"].to(modelo.device)
            batch_labels = batch["label"].to(modelo.device) 

            # TODO Zero grad, forward pass, backward pass, optimizer step
            optimizer.zero_grad()
            logits, proba = modelo(batch_imgs)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

            # TODO acumula el costo
            train_loss += loss.item()
            train_correct += calculate_accuracy(logits, batch_labels)
            train_total += batch_labels.size(0)

            loop.set_postfix(loss=loss.item())

        # TODO Calcula el costo promedio
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = (train_correct / train_total) * 100

        val_loss, val_acc = validation_step(val_loader, modelo, criterion)
        scheduler.step(val_loss)

        tqdm.write(
            f"Epoch {epoch+1}: "
            f"Train Loss: {avg_train_loss:.4f} | Acc: {avg_train_acc:.2f}% || "
            f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%"
        )

        # TODO guarda el modelo si el costo de validación es menor al mejor costo de validación
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc

            tqdm.write(f"Nuevo mejor modelo ({best_val_loss:.4f}). Guardando localmente...")
            modelo.save_model(new_model_filename)

        run.log({
            "epoch": epoch,
            "train/loss": avg_train_loss,
            "train/accuracy": avg_train_acc,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
    
    print(f"Entrenamiento finalizado. Mejor Val Accuracy: {best_val_acc:.2f}%")

    print(f"Subiendo el mejor modelo a WandB...")
    artifact = wandb.Artifact(
        name=f"fer2013-resnet18-{run_type}", 
        type="model",
        description=f"Mejor modelo final. Loss {best_val_loss:.4f}, Acc> {best_val_acc:.2f}%"
    )
    path_to_model = file_path / "models" / new_model_filename
    if path_to_model.exists():
        artifact.add_file(path_to_model)
        run.log_artifact(artifact)
        print("Modelo subido exitosamente!")
    else:
        print("No se encontró el archivo del modelo para subir.")
    
    wandb.finish()


if __name__ == "__main__":    
    train(
        n_epochs=100, 
        learning_rate=1e-5,
        batch_size=256, 
        freeze_backbone=False,
        pretrained=True,
        base_model_path="best_model_2025-11-17_16-35-31-137208.pth"
    )
