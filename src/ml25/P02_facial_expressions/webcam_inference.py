import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from network import Network
import pathlib
import time
from dataset import EMOTIONS_MAP

# --- CONFIGURACIÓN ---
MODEL_PATH = pathlib.Path(__file__).parent / "models" / "best_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_trained_model():
    print(f"Cargando modelo en {DEVICE}...")
    if not MODEL_PATH.exists():
        print(f"ERROR: No se encontró el modelo en: {MODEL_PATH}")
        print("Asegúrate de haber corrido training.py primero.")
        exit()

    # Inicializar la arquitectura (48x48 input, 7 clases)
    model = Network(input_dim=48, n_classes=7)
    
    # Cargar los pesos
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except Exception as e:
        print(f"Error cargando los pesos: {e}")
        exit()
        
    model.to(DEVICE)
    model.eval() # Modo evaluación (apaga Dropout)
    print("✅ Modelo cargado exitosamente.")
    return model

def preprocess_face(face_img_gray):
    """
    Convierte el recorte de opencv (numpy) a tensor listo para la red
    """
    # 1. Convertir a PIL o trabajar directo con tensor
    # Simplemente convertimos a tensor y redimensionamos
    img_tensor = torch.from_numpy(face_img_gray)
    
    # 2. Normalizar a [0, 1] (OpenCV da 0-255)
    img_tensor = img_tensor.float() / 255.0
    
    # 3. Redimensionar a 48x48 (usando interpolate requiere 4 dimensiones)
    # [H, W] -> [1, 1, H, W] para interpolate
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
    
    img_tensor = torch.nn.functional.interpolate(img_tensor, size=(48, 48), mode='bilinear')
    
    # El modelo espera [Batch, Channels, H, W] -> [1, 1, 48, 48]
    # Ya tiene esa forma por el paso anterior
    
    return img_tensor.to(DEVICE)

def main():
    # 1. Cargar Modelo
    net = load_trained_model()

    # 2. Cargar Detector de Rostros (Haar Cascade)
    # OpenCV incluye esto por defecto
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 3. Iniciar Webcam (0 = cámara default)
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("❌ No se pudo abrir la webcam.")
        return

    print("\nWEBCAM INICIADA")
    print(" - Encuadra tu cara.")
    print(" - Presiona 'q' para salir.\n")

    prev_frame_time = 0
    new_frame_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Voltear horizontalmente para efecto espejo (más natural)
        frame = cv2.flip(frame, 1)
        
        # Convertir a escala de grises para el detector de rostros
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar caras
        faces = face_cascade.detectMultiScale(
            gray_frame, 
            scaleFactor=1.3, 
            minNeighbors=5, 
            minSize=(30, 30)
        )

        for (x, y, w, h) in faces:
            # Dibujar recuadro
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

            # --- INFERENCIA ---
            # Recortar solo la cara
            roi_gray = gray_frame[y:y+h, x:x+w]
            
            if roi_gray.size == 0:
                continue

            # Preprocesar y Predecir
            tensor_input = preprocess_face(roi_gray)
            
            with torch.inference_mode():
                logits, proba = net(tensor_input)
                # Obtener la clase con mayor probabilidad
                pred_idx = torch.argmax(proba, dim=1).item()
                confidence = proba[0][pred_idx].item() * 100
                label = EMOTIONS_MAP[pred_idx]

            # --- DIBUJAR RESULTADO ---
            label_text = f"{label}: {confidence:.1f}%"
            
            # Color dinámico: Rojo si es negativo, Verde si es positivo
            color = (0, 255, 0) 
            if label in ["Enojo", "Disgusto", "Miedo", "Tristeza"]:
                color = (0, 0, 255)
            elif label == "Neutral":
                color = (255, 255, 255)

            # Poner fondo negro al texto para leerlo mejor
            cv2.rectangle(frame, (x, y-35), (x+w, y), color, -1)
            cv2.putText(frame, label_text, (x + 5, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # Calcular FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Mostrar video
        cv2.imshow('Detector de Emociones (ResNet18)', frame)

        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()