# 🔍 Detección de Fraude Financiero mediante Aprendizaje Automático

> **Proyecto Final — Modelos 2 | Universidad de Antioquia, 2026**  
> Juan Mosquera · María Mosquera · Michael Molina  
> Departamento de Ingeniería de Sistemas, Facultad de Ingeniería

---

## 📌 Resumen Ejecutivo

Sistema de detección de fraude financiero basado en técnicas de **aprendizaje automático supervisado**, desarrollado sobre el dataset público **IEEE-CIS Fraud Detection** (Kaggle). El proyecto abarcó desde el preprocesamiento y análisis exploratorio hasta la optimización de hiperparámetros y análisis de reducción de dimensionalidad, logrando un modelo final con **ROC-AUC de 0.9500** y **PR-AUC de 0.7943**.

> Enlace video Youtube - https://youtu.be/_eGB1w32rek

---

## 📂 Estructura del Repositorio

```
Modelos-2---proyecto-de-aula/
│
├── py_Juan_Mosquera_Maria_Mosquera_Michael_Molina_01_fraud_detection.ipynb
│   └── Notebook principal con todo el pipeline experimental
│
├── py_Juan_Mosquera_Maria_Mosquera_Michael_Molina.pdf
│   └── Artículo académico con resultados y análisis completo
│
└── README.md
    └── Este archivo
```

---

## 🗄️ Dataset

| Atributo | Detalle |
|---|---|
| Fuente | [IEEE-CIS Fraud Detection — Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection/data) |
| Registros totales | 155,504 |
| Registros usados (post-merge) | 106,239 |
| Variables originales | 633 |
| Variable objetivo | `isFraud` (0 = legítima, 1 = fraude) |
| Desbalance de clases | Fuerte — fraude es clase minoritaria |

El dataset contiene información transaccional, características de comportamiento de compra, datos de dispositivos e información de identidad anonimizada.

---

## 🔬 Metodología

El pipeline experimental fue organizado en **cinco etapas secuenciales**:

```
1. Preprocesamiento y EDA
       ↓
2. Auditoría y Selección de Características
       ↓
3. Entrenamiento y Evaluación de Modelos
       ↓
4. Optimización de Hiperparámetros (XGBoost)
       ↓
5. Reducción de Dimensionalidad (PCA / UMAP)
```

### 1️⃣ Preprocesamiento y EDA

- Imputación de valores faltantes con la **mediana** (robustez ante distribuciones asimétricas)
- Codificación numérica de variables categóricas
- Análisis exploratorio bivariado de la variable objetivo

### 2️⃣ Auditoría y Selección de Características

| Etapa | Variables eliminadas | Variables restantes |
|---|---|---|
| Conjunto inicial | — | 633 |
| Baja varianza (< 0.01) | 118 | 515 |
| Alta correlación (> 0.95) | 276 | **239** |

Reducción acumulada del **62.2%** respecto al espacio original.

Adicionalmente, se construyó un **ranking consolidado** de importancia combinando tres métodos:
- 📊 Mutual Information
- 🌲 Random Forest Feature Importance
- ⚡ XGBoost Feature Importance

### 3️⃣ Entrenamiento y Evaluación de Modelos

Se evaluaron seis algoritmos de clasificación supervisada con partición estratificada 80/20:

| Modelo | ROC-AUC | PR-AUC | Precisión | Recall | F1 |
|---|---|---|---|---|---|
| **XGBoost** | **0.9285** | **0.7024** | 0.3575 | 0.7383 | 0.4817 |
| MLPClassifier | 0.8570 | 0.5371 | 0.4158 | 0.5402 | 0.4699 |
| Random Forest | 0.8958 | 0.5174 | 0.1949 | 0.7215 | 0.3068 |
| Decision Tree | 0.8640 | 0.3810 | 0.1438 | 0.7664 | 0.2421 |
| Logistic Regression | 0.8651 | 0.3532 | 0.0973 | 0.7589 | 0.1725 |
| SVC | 0.5472 | 0.0718 | 0.0247 | 0.8804 | 0.0481 |

> **Métrica principal:** PR-AUC — más representativa en problemas con fuerte desbalance de clases.

### 4️⃣ Optimización de XGBoost

Búsqueda aleatoria de hiperparámetros (`RandomizedSearchCV`) con **validación cruzada estratificada de 5 folds**.

| Hiperparámetro | Valor Óptimo |
|---|---|
| `max_depth` | 8 |
| `learning_rate` | 0.136 |
| `gamma` | 0.184 |
| `colsample_bytree` | 0.891 |
| `n_estimators` | 49 |

**Mejora obtenida:**

| Métrica | Baseline | Optimizado | Δ |
|---|---|---|---|
| ROC-AUC | 0.9285 | **0.9500** | +2.3% |
| PR-AUC | 0.7024 | **0.7943** | +~13% |
| Precisión (fraude) | — | 0.8903 | — |
| Recall (fraude) | — | 0.6673 | — |
| F1-Score (fraude) | — | 0.7628 | — |

### 5️⃣ Reducción de Dimensionalidad

| Estrategia | Variables | Reducción | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| XGBoost Tuned (239 var.) | 239 | 0% | **0.9500** | **0.7943** |
| XGBoost + PCA (110 comp.) | 110 | 54.0% | 0.8957 | 0.5799 |
| XGBoost + UMAP (2 comp.) | 2 | 99.2% | 0.7696 | 0.1860 |

> **Conclusión:** Las 239 características seleccionadas representan el espacio **óptimo** para el problema. Ni PCA ni UMAP lograron superar el rendimiento del modelo sin reducción adicional.

---

## 🏆 Resultados Finales — Modelo XGBoost Optimizado

```
ROC-AUC  → 0.9500
PR-AUC   → 0.7943
Precisión → 0.8903
Recall   → 0.6673
F1-Score → 0.7628
```

**Matriz de Confusión:**

|  | Predicho: No Fraude | Predicho: Fraude |
|---|---|---|
| **Real: No Fraude** | 20,669 ✅ | 44 ❌ |
| **Real: Fraude** | 178 ❌ | 357 ✅ |

---

## ⚙️ Configuración del Entorno

### Dependencias principales

```python
# Machine Learning
xgboost
scikit-learn
imbalanced-learn        # SMOTE

# Reducción de dimensionalidad
umap-learn

# Manipulación de datos
pandas
numpy

# Visualización
matplotlib
seaborn
```

### Instalación

```bash
pip install xgboost scikit-learn imbalanced-learn umap-learn pandas numpy matplotlib seaborn
```

### Ejecución

```bash
jupyter notebook py_Juan_Mosquera_Maria_Mosquera_Michael_Molina_01_fraud_detection.ipynb
```

> ⚠️ Se recomienda usar `random_state=42` para reproducibilidad. El notebook ya lo implementa en todos los procesos de partición, entrenamiento y validación.

---

## 📊 Decisiones de Diseño Clave

### ¿Por qué PR-AUC como métrica principal?
En problemas con fuerte desbalance de clases (como detección de fraude), la exactitud (_accuracy_) es engañosa. PR-AUC evalúa simultáneamente la precisión y la sensibilidad sobre la **clase minoritaria**, proporcionando una medida más representativa del rendimiento real del clasificador.

### ¿Por qué XGBoost?
Los modelos basados en _gradient boosting_ demuestran sistemáticamente mayor capacidad para capturar relaciones no lineales en datos tabulares de alta dimensionalidad. La literatura del estado del arte (Almalki & Masud, 2025; IEEE-CIS XGB, 2024) respalda esta elección.

### ¿Por qué no SMOTE como estrategia principal?
Se evaluó SMOTE dentro de pipelines para modelos sensibles al desbalance (Regresión Logística, MLP). Para XGBoost, se utilizó el parámetro `scale_pos_weight` que maneja el desbalance de forma nativa y más eficiente para este tipo de modelos.

---

## 📚 Referencias Seleccionadas

1. Dal Pozzolo et al. — *Calibrating probability with undersampling for unbalanced classification* (CIDM, 2015)
2. Almalki & Masud — *Financial Fraud Detection Using Explainable AI and Stacking Ensemble Methods* ([arXiv:2505.10050](https://arxiv.org/abs/2505.10050), 2025)
3. Kaggle — [IEEE-CIS Fraud Detection Dataset](https://www.kaggle.com/competitions/ieee-fraud-detection/data) (2019)
4. ResearchGate — [IEEE-CIS Fraud Detection Based on XGB](https://www.researchgate.net/publication/378508394) (2024)

---

## 👥 Autores

| Nombre | Correo |
|---|---|
| Juan Mosquera | juan.mosquerap@udea.edu.co |
| María Mosquera | mpaula.mosquera@udea.edu.co |
| Michael Molina | michael.molina@udea.edu.co |

**Universidad de Antioquia** — Departamento de Ingeniería de Sistemas, Facultad de Ingeniería  
Medellín, Colombia · Mayo 2026

---

<p align="center"><i>Proyecto Final · Modelos 2 · 2026</i></p>
