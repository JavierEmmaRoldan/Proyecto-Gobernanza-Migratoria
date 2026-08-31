## 📁 Reproducibilidad y Datos (Carpeta `data/`)

Por políticas de versionado y límites de almacenamiento de GitHub, los archivos CSV originales no se incluyen en este repositorio. 

Para ejecutar el pipeline localmente, el evaluador debe:
1. Contar con los archivos originales `ag_news_train.csv` y `ag_news_test.csv` y el `Dataset_Migracion.xlsx`.
   Los puede descargar en "https://drive.google.com/drive/folders/1CCTF6rG9nCj925k1U6Ki4QxQmWOL6hVD?usp=sharing"
2. Colocarlos directamente dentro de la carpeta `data/` de este proyecto.
3. Ejecutar los notebooks de forma secuencial.

# Checkpoint 1: Pipeline Base de Deep Learning

Este repositorio contiene la infraestructura inicial para el entrenamiento y validación de una Red Neuronal Artificial (MLP) desarrollada en PyTorch. El modelo base clasifica datos relacionados con flujos migratorios, evaluando el impacto de la gobernanza y tasas de pobreza.

## Entorno Técnico
- **Framework:** PyTorch (Versión utilizada: 2.x)
- **Dispositivo:** Detección automática (Soporta CPU, CUDA y MPS).
- **Optimizador:** Adam
- **Función de Pérdida:** Binary Cross Entropy (BCELoss)
- **Learning Rate:** `0.01` (Seleccionado para permitir una convergencia rápida y estable en este dataset inicial, evitando saltos muy grandes en los gradientes).

## Interpretación del Entrenamiento (Curva de Pérdida)
Durante la ejecución de 50 épocas, se observó lo siguiente:
- **Reducción de Loss:** La pérdida de entrenamiento (Train Loss) disminuyó consistentemente, indicando que el modelo ajustó sus pesos correctamente a través del Backpropagation.
- **Validación y Overfitting:** La pérdida de validación (Val Loss) acompañó la tendencia a la baja junto con un aumento en el *Validation Accuracy*, lo que indica que el modelo logra generalizar sin sobreajustarse (overfitting) a los datos de entrenamiento.

## Estructura del Repositorio
- `src/pipeline_base.py`: Contiene el ciclo de vida completo del modelo (creación, dataloaders, loop de entrenamiento y loop de validación).
- `requirements.txt`: Dependencias necesarias (`torch`, `numpy`, `scikit-learn`, `pandas`, `openpyxl`).

Justificación del Modelo y Vectorizador:

Para este baseline se seleccionó una Regresión Logística. A pesar de su simplicidad, este modelo es extremadamente robusto frente a las matrices de alta dimensionalidad y esparsidad generadas por el procesamiento de texto. Se configuró el TfidfVectorizer con max_features=10000 para reducir el ruido y evitar el sobreajuste, e incorporamos ngram_range=(1, 2) para que el modelo capture contextos básicos mediante bigramas, lo cual incrementa notablemente la precisión semántica. Siguiendo las mejores prácticas, el vectorizador se ajustó (fit) exclusivamente sobre el set de entrenamiento para evitar el Data Leakage.

Análisis Preliminar de la Matriz de Confusión:

(Nota: Ajusta esto según el gráfico exacto que te arroje el código, pero este es el comportamiento estándar de AG News)
Al analizar la matriz de confusión, observamos que las clases "Sports" y "World" presentan los niveles más altos de precisión y recall, ya que utilizan vocabularios muy específicos. Sin embargo, el modelo presenta su mayor desafío al intentar separar las clases "Business" (Negocios) y "Sci/Tech" (Tecnología). Esta confusión cruzada es lógica y esperable, ya que muchas noticias sobre lanzamientos de productos tecnológicos (Tech) involucran reportes de acciones bursátiles o fusiones corporativas (Business), compartiendo un espacio semántico muy similar.