# Portafolio de Deep Learning y NLP

## 📁 Reproducibilidad y Datos (Carpeta `data/`)
Por políticas de versionado y límites de almacenamiento de GitHub, los archivos CSV y Excel originales no se incluyen en este repositorio. 

Para ejecutar los pipelines localmente, el evaluador debe:
1. Contar con los archivos originales `ag_news_train.csv`, `ag_news_test.csv` y el `Dataset_Migracion.xlsx`.
   Se pueden descargar de forma segura desde aquí: [Google Drive - Datasets del Proyecto](https://drive.google.com/drive/folders/1CCTF6rG9nCj925k1U6Ki4QxQmWOL6hVD?usp=sharing)
2. Colocarlos directamente dentro de la carpeta `data/` de este proyecto.
3. Ejecutar los notebooks de forma secuencial.

---

## ⚙️ Tecnologías e Instalación
Este proyecto utiliza un entorno basado en Python 3.x. Las principales librerías involucradas abarcan desde Machine Learning clásico (`scikit-learn`) hasta Deep Learning avanzado (`PyTorch`, `transformers`, `peft`).

Para instalar todas las dependencias necesarias de una sola vez, ejecuta el siguiente comando en tu terminal:
```bash
pip install -r requirements.txt

## 📌 Checkpoint 1: Pipeline Base de Deep Learning (Datos Tabulares)
Este módulo contiene la infraestructura inicial para el entrenamiento y validación de una Red Neuronal Artificial (MLP) desarrollada en PyTorch. El modelo base clasifica datos relacionados con flujos migratorios, evaluando el impacto de la gobernanza y tasas de pobreza.

* **Entorno Técnico:** PyTorch (Versión 2.x), Optimizador Adam, Función de Pérdida BCELoss.
* **Learning Rate:** `0.01` (Seleccionado para permitir una convergencia rápida y estable).
* **Análisis de Entrenamiento:** Durante la ejecución de 50 épocas, la pérdida de entrenamiento (Train Loss) y validación (Val Loss) disminuyeron consistentemente, indicando que el modelo logra generalizar sin sobreajustarse (overfitting).

---

## 📌 Checkpoint 2: Baseline de NLP Clásico (AG News)
Se construyó un modelo base para clasificación de texto utilizando técnicas de Machine Learning clásico sobre el dataset de noticias AG News.

* **Justificación del Modelo y Vectorizador:** Para este baseline se seleccionó una Regresión Logística. A pesar de su simplicidad, este modelo es extremadamente robusto frente a matrices de alta dimensionalidad. Se configuró el `TfidfVectorizer` con `max_features=10000` y `ngram_range=(1, 2)` para capturar contextos básicos mediante bigramas. El vectorizador se ajustó exclusivamente sobre el set de entrenamiento para evitar el Data Leakage.
* **Análisis de la Matriz de Confusión:** Observamos que las clases "Sports" y "World" presentan los niveles más altos de precisión. Sin embargo, el modelo presenta su mayor desafío al separar las clases "Business" y "Sci/Tech". Esta confusión cruzada es lógica, ya que muchas noticias sobre tecnología involucran reportes bursátiles o corporativos, compartiendo un espacio semántico similar.

---

## 📌 Checkpoint 3: Fine-Tuning de Transformers con LoRA (AG News)
Para superar la barrera semántica del baseline clásico, se implementó un modelo de Deep Learning avanzado utilizando el ecosistema de Hugging Face.

* **Arquitectura:** Se utilizó `DistilBERT-base-uncased`, logrando procesar el contexto de las oraciones completas gracias al mecanismo de Auto-Atención.
* **Eficiencia Paramétrica (PEFT):** Para evitar el costo prohibitivo de entrenar toda la red, se aplicó la técnica **LoRA (Low-Rank Adaptation)** inyectando adaptadores en las matrices de atención (`q_lin` y `v_lin`) con un rango `r=8`.
* **Impacto:** Esta técnica permitió entrenar menos del 1% de los parámetros totales del modelo en Google Colab, reduciendo el solapamiento entre las clases "Business" y "Sci/Tech" y superando las métricas obtenidas por la Regresión Logística clásica.