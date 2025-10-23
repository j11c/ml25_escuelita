# Proyecto: Predicción de Compra

## Descripción
Este proyecto predice si un cliente comprará un nuevo producto. La definición del proyecto se puede ver en el readme [`definicion`](./DEFINICION.md)
El flujo está dividido en tres pasos: **exploración**, **entrenamiento** y **generar CSV de predicciones**.

---

## Archivos principales

### 1️. Exploración de datos
Notebook para analizar el dataset y generar gráficos.
[notebook](./data_exploration.py)

### 2. Entrenamiento
script para entrenar el modelo y correr experimentos. Ejecutar el siguiente archivo:
[training.py](./training.py)

O en la consola como ...
```
conda activate mlenv
python training.py
```

### 13. Resultados
script para generar el csv que se sube a la competencia de kaggle. Ejecutar el siguiente archivo:
[inference.py](./inference.py)