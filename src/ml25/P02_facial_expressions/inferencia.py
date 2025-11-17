import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from network import Network
import torch
from utils import (
    to_numpy,
    get_transforms,
    add_img_text,
)
from dataset import EMOTIONS_MAP
import pathlib

file_path = pathlib.Path(__file__).parent.absolute()

# Configuracion
MODEL_NAME = "best_model_2025-11-17_06-25-07-652772.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def detect_face(img):
    """
    Intenta detectar una cara en la imagen.
    - Si encuentra cara, regresa el recorte (ROI).
    - Si no encuentra, regresa la imagen original (asume que ya está recortada).
    """
    # Convertir a gris para el detector
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Cargar Haar Cascade (viene incluido en cv2)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detección
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) > 0:
        print(f"Rostro detectado. Recortando...")
        x, y, w, h = faces[0]
        return img[y:y+h, x:x+w]
    else:
        print("No se detectó rostro (o la imagen ya es un recorte). Usando imagen completa.")
        return img


def load_img(path):
    assert os.path.isfile(path), f"El archivo {path} no existe"
    img = cv2.imread(path)

    if img is None:
        raise ValueError(f"No se pudo leer la imagen {path}")
    
    face_img = detect_face(img)
    val_transforms, unnormalize = get_transforms("test", img_size=48)
    tensor_img = val_transforms(face_img)

    denormalized = unnormalize(tensor_img)

    if len(tensor_img.shape) == 3:
        tensor_img = tensor_img.unsqueeze(0)

    return img, tensor_img, denormalized


def predict(img_title_paths):
    """
    Hace la inferencia de las imagenes
    args:
    - img_title_paths (dict): diccionario con el titulo de la imagen (key) y el path (value)
    """
    print(f"Cargando modelo desde: models/{MODEL_NAME} en {DEVICE}")

    # Cargar el modelo
    modelo = Network(48, 7)
    model_path = file_path / "models" / MODEL_NAME

    try:
        modelo.load_state_dict(torch.load(model_path, map_location=DEVICE))
    except Exception as e:
        print(f"Error cargando modelo: {e}")
        return

    modelo.to(DEVICE)
    modelo.eval() # Importante para desactivar Dropout

    for path in img_title_paths:
        full_path = (file_path / path).as_posix()
        print(f"Procesando: {full_path}")

        try:
            # AQUI RECIBIMOS LAS 3 VARIABLES
            original_crop, transformed, denormalized = load_img(full_path)
            
            # Mover a GPU
            transformed = transformed.to(DEVICE)

            # Inferencia
            logits, proba = modelo.predict(transformed)
            pred_idx = torch.argmax(proba, -1).item()
            pred_label = EMOTIONS_MAP[pred_idx]
            confidence = proba[0][pred_idx].item() * 100

            # --- VISUALIZACIÓN ---
            h, w = original_crop.shape[:2]
            resize_value = 300
            
            # 1. Imagen Original
            display_img = cv2.resize(original_crop, (w * resize_value // h, resize_value))
            display_img = add_img_text(display_img, f"{pred_label}: {confidence:.1f}%")

            # 2. Imagen Transformada (Denormalized ya viene calculada)
            # Solo necesitamos pasarla a Numpy y cambiar canales si es necesario
            denorm_np = to_numpy(denormalized)
            denorm_np = cv2.resize(denorm_np, (resize_value, resize_value))

            cv2.imshow("Prediccion - Recorte Original", display_img)
            cv2.imshow("Prediccion - Input Red", denorm_np)
            
            if cv2.waitKey(0) == ord('q'):
                break

        except Exception as e:
            print(f"Error: {e}")
            continue
    
    cv2.destroyAllWindows()

    # for path in img_title_paths:
    #     # Cargar la imagen
    #     # np.ndarray, torch.Tensor
    #     im_file = (file_path / path).as_posix()
    #     original, transformed, denormalized = load_img(im_file)

    #     # Inferencia
    #     logits, proba = modelo.predict(transformed)
    #     pred = torch.argmax(proba, -1).item()
    #     pred_label = EMOTIONS_MAP[pred]

    #     # Original / transformada
    #     h, w = original.shape[:2]
    #     resize_value = 300
    #     img = cv2.resize(original, (w * resize_value // h, resize_value))
    #     img = add_img_text(img, f"Pred: {pred_label}")

    #     # Mostrar la imagen
    #     denormalized = to_numpy(denormalized)
    #     denormalized = cv2.resize(denormalized, (resize_value, resize_value))
    #     cv2.imshow("Predicción - original", img)
    #     cv2.imshow("Predicción - transformed", denormalized)
    #     cv2.waitKey(0)


if __name__ == "__main__":
    # Direcciones relativas a este archivo
    img_paths = [
        "./test_imgs/happy.png",
        "./test_imgs/sad.jpg",
        "./test_imgs/angry.jpg",
        "./test_imgs/joshua_surprised.jpg",
        "./test_imgs/joshua_sad.jpg"
    ]
    predict(img_paths)
