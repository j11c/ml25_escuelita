# Proyecto: Predicción de Compras de Nuevos Ítems

## Objetivo
Dado un historial de compras de clientes y las características de los productos, el objetivo del proyecto es **predecir si un cliente existente comprará un nuevo producto** que será lanzado próximamente dentro de los siguientes 30 días.

La fecha al día de recolección de datos fue: 21, sept. 2025
---

## Entregables

1. **Exploración de datos (requerido)**
   - Pueden usar el notebook `data_exploration.ipynb` para complementar su exploración de datos. Correr el notebook no es suficiente, tienen que hacer exploración adicional.
   - Revisar valores faltantes, tipos de datos, distribuciones de features.
   - Analizar relaciones entre clientes, productos y compras.

2. **Pipeline de ML (requerido)**: Traten de mantener su código organizado de la siguiente manera
   - Scripts:
     - `data_processing.py`: extracción, preprocesamiento e ingenieria de atributos.
     - `model.py`: definición de modelo y pipeline.
     - `training.py`: entrenamiento del modelo.
     - `inference.py`: predicciones sobre nuevos productos.
   - Implementar generación de ejemplos negativos (clientes que no compraron un ítem).
   - Dividir datos en train/test, se aconseja considerar cold-start para productos nuevos.

3. **Monitoreo y resultados(requerido)**
   - Pueden usar el logger provisto (`utils.py`)
   - Se recomienda, para todo experimento, registrar:
     - Métricas de evaluación y resultados
     - Hyperparámetros del modelo entrenado/evaluado
   - Presentar un análisis de resultados con diferentes métricas y métodos/modelos utilizados
   - Experimentos con distintos modelos o embeddings de usuarios/ítems.

4. **Opcional / Bonus**
   - Análisis exploratorio con visualizaciones avanzadas.
   - Análisis de resultados extensivo con múltiples visualizaciones
   - Equipo con mejores métricas en la competencia de kaggle

5. ** Presentación **
El proyecto será evaluado en dos etapas adicionales al código.

1. Como el 50% del primer examen parcial
2. En una presentación corta de 10 min máximo por equipo, presentando su solución con enfoque a las decisiones de diseño, resultados e interpretación de los mismos.

   - La presentación deberá estar orientada a alguien **sin conocimiento técnico en el tema** pero con conocimiento general de programación (por ejemplo, un compañero de otra carrera).  
   - El objetivo de la presentación es evaluar su capacidad de **enseñar conceptos técnicos a un público no técnico** y justificar sus resultados.  
   - Evitar mostrar código, a menos que sea una pieza muy importante y relevante para su solución. Priorizar diagramas, gráficos y visualizaciones.
   - **Conclusión obligatoria:** deben incluir un resumen conciso de lo realizado en el proyecto y especificar cuál solución consideran mejor, justificando su decisión de manera **técnica** (por ejemplo, elección de atributos, modelo, métricas etc.).

---


## Definición del problema
El objetivo del proyecto es predecir la probabilidad de que un cliente existente compre un nuevo ítem de moda que será lanzado próximamente, basándose en el historial de compras anteriores y las características de los productos.

## Datos disponibles

### Training Dataset
Contiene información histórica de compras y características de los clientes y productos:
- purchase_id: Identificador único de compra
- customer_id: identificador único del cliente
- customer_date_of_birth: Fecha de nacimiento del usuario
- customer_gender: género del cliente (male / female)
- customer_age: edad del cliente
- customer_signup_date: Fecha en la que se registró el cliente
- item_id: identificador único del ítem comprado
- item_title: título del producto
- item_category: categoría del producto (t-shirt, blouse, dress, shoes, skirt, jeans, suit)
- item_price: precio del producto
- item_img_filename: nombre del archivo de la imagen del producto
- item_avg_rating: Calificación promedio del producto
- item_num_ratings: Cantidad de reseñas del producto
- item_release_date: fecha de lanzamiento del producto
- customer_item_views: Vistas previas del usuario al producto
- purchase_item_rating: Calificación que el usuario otorgó al producto
- purchase_device: Dispositivo por el cual se hizo la compra
- purchase_timestamp: fecha de compra
- label: etiqueta


### Test Dataset
Contiene información de nuevos productos * que aún no han sido lanzados* (su fecha de lanzamiento es posterior a sept. 21, 2025). Su tarea es decidir si un usuario existente comprará un producto que será lanzado próximamente.

- purchase_id: Identificador único de compra
- customer_id: identificador único del cliente
- customer_date_of_birth: Fecha de nacimiento del usuario
- customer_gender: género del cliente (male / female)
- customer_age: edad del cliente
- customer_signup_date: Fecha en la que se registró el cliente
- item_id: identificador único del ítem comprado
- item_title: título del producto
- item_category: categoría del producto (t-shirt, blouse, dress, shoes, skirt, jeans, suit)
- item_price: precio del producto
- item_img_filename: nombre del archivo de la imagen del producto
- item_avg_rating: Calificación promedio del producto
- item_num_ratings: Cantidad de reseñas del producto
- item_release_date: fecha de lanzamiento del producto
- customer_item_views: Vistas previas del usuario al producto
- purchase_item_rating: Calificación que el usuario otorgó al producto
- purchase_device: Dispositivo por el cual se hizo la compra
- purchase_timestamp: fecha de compra

## Tarea principal
Para cada nuevo producto en el conjunto de test, se debe predecir si un cliente existente comprará dicho producto.
- Objetivo primario: clasificación binaria (compra / no compra)
- Objetivo opcional: Predecir la probabilidad de compra (likelihood) para cada cliente-producto.


## Consideraciones
- Se espera que manejen el cold-start problem de los productos nuevos, es decir, que el modelo no tenga historial de compra para esos productos.
- Se pueden utilizar tanto atributos del cliente como del producto para la predicción.
- Se incentiva la ingeniería de atributos y preprocesamiento de datos.

## Recomendaciones
- Siempre explorar los datos antes de entrenar modelos.
- Asegúrate de documentar tus decisiones y hallazgos de manera ordenada.
- Mantener scripts y funciones modulares, reutilizables y limpias.

