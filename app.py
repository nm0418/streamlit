"""Prototipo académico: clasificación del nivel de riesgo (dataset sintético de salud) con regresión logística."""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from health_utils import (CATEGORIES, CLASS_ORDER, FEATURES, LABELS, MES_KEYS, RANGES, TARGET,
                          record_to_frame, validate_record)

# ----------------------------------------------------------------------------- configuración
INTEGRANTES = ["Nelsy Madera"]  # <-- agregar aquí los nombres completos de todos los integrantes
ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "pipeline_salud.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
CLEAN_PATH = ROOT / "data" / "datos_limpios.csv"
TEST_PATH = ROOT / "data" / "test_set.csv"

st.set_page_config(page_title="Nivel de riesgo en salud (prototipo)", page_icon="🩺", layout="wide")


@st.cache_resource
def load_pipeline():
    """Carga el pipeline guardado (preprocesamiento + modelo). No entrena en cada ejecución."""
    try:
        return joblib.load(MODEL_PATH)
    except Exception as exc:  # archivo ausente o versión distinta de scikit-learn
        st.warning(f"No se pudo cargar el modelo guardado ({type(exc).__name__}). "
                   "Se regenera una sola vez con train_model.py (requiere data/Dataset_Sintetico_Salud_.xlsx).")
        from train_model import build_and_save
        return build_and_save()[0]


pipe = load_pipeline()
try:
    df = pd.read_csv(CLEAN_PATH, index_col="id_original")
    test_df = pd.read_csv(TEST_PATH, index_col="id_original")
    metrics = json.loads(METRICS_PATH.read_text())
except FileNotFoundError as exc:
    st.error(f"Falta un archivo del repositorio: {exc.filename}. Ejecuta `python train_model.py` y súbelo a GitHub.")
    st.stop()

# ----------------------------------------------------------------------------- encabezado
st.title("🩺 Clasificación del nivel de riesgo en salud")
st.caption("Prototipo académico con datos **sintéticos**. No sustituye la evaluación ni la decisión de un "
           "profesional de la salud y no tiene validez clínica.")
tab_proy, tab_eda, tab_pred, tab_mod = st.tabs(["Proyecto", "Análisis exploratorio", "Predicción", "Modelo y limitaciones"])

# ----------------------------------------------------------------------------- 1. Proyecto
with tab_proy:
    st.subheader("Presentación")
    st.markdown(
        "**Proyecto:** comparación de técnicas de minería de datos (regresión lineal, regresión logística, "
        "árbol de decisión y K-means) sobre un conjunto sintético de salud. Esta aplicación despliega el "
        "mejor clasificador: la **regresión logística**.\n\n"
        "**Objetivo de la aplicación:** estimar el nivel de riesgo (**Bajo, Medio o Alto**) a partir de 12 "
        "características de un registro y mostrar qué tan confiable es el modelo.\n\n"
        "**Curso:** Minería de Datos – Corporación Universitaria de la Costa (CUC)."
    )
    st.markdown("**Integrantes:** " + ", ".join(INTEGRANTES))
    st.subheader("Datos")
    st.markdown(
        f"Conjunto sintético de salud (`Dataset_Sintetico_Salud_.xlsx`): 500 registros originales y 14 columnas; "
        f"tras eliminar 3 duplicados exactos quedan {metrics['n_total']} registros. Hay algunos valores faltantes "
        "en IMC, glucosa, colesterol, actividad física y horas de sueño, que el pipeline imputa. "
        "`ID_Paciente` no se usa como predictor. Al ser sintético, los resultados son solo educativos."
    )
    st.dataframe(df.head(10))

# ----------------------------------------------------------------------------- 2. EDA
with tab_eda:
    st.subheader("Visualizaciones del análisis exploratorio")
    colors = {"Bajo": "#59A14F", "Medio": "#EDC948", "Alto": "#E15759"}
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5, 3.6))
        counts = df[TARGET].value_counts().reindex(CLASS_ORDER)
        ax.bar(counts.index, counts.values, color=[colors[c] for c in counts.index])
        for i, v in enumerate(counts.values):
            ax.text(i, v + 3, f"{v} ({v / len(df):.1%})".replace(".", ","), ha="center")
        ax.set_ylabel("Registros")
        ax.set_title("Figura 1. Distribución del nivel de riesgo")
        st.pyplot(fig)
        st.caption("Medio es la clase más frecuente y Alto la menos frecuente; hay un desbalance moderado.")
    with c2:
        fig, ax = plt.subplots(figsize=(5, 3.6))
        ax.boxplot([df.loc[df[TARGET] == c, "Glucosa_mg_dL"].dropna() for c in CLASS_ORDER], tick_labels=CLASS_ORDER)
        ax.set_ylabel("Glucosa (mg/dL)")
        ax.set_title("Figura 2. Glucosa según nivel de riesgo")
        st.pyplot(fig)
        st.caption("La glucosa tiende a aumentar con el nivel de riesgo. Ojo: el nivel de riesgo pudo construirse "
                   "con variables clínicas como esta, por lo que la relación puede ser en parte circular.")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, col in zip(axes, ["Edad", "IMC"]):
        ax.boxplot([df.loc[df[TARGET] == c, col].dropna() for c in CLASS_ORDER], tick_labels=CLASS_ORDER)
        ax.set_ylabel(LABELS[col])
    fig.suptitle("Figura 3. Edad e IMC según nivel de riesgo")
    st.pyplot(fig)
    st.caption("El riesgo alto se asocia con mayor edad e IMC. Son asociaciones descriptivas, no causales.")

# ----------------------------------------------------------------------------- 3. Predicción
DEFAULTS = {"Edad": 45, "Genero": "Femenino", "IMC": 26.0, "Glucosa_mg_dL": 100.0, "Presion_Sistolica_mmHg": 120,
            "Colesterol_mg_dL": 190.0, "Actividad_Fisica": "Moderada", "Fumador": "No", "Horas_Sueno": 7.0,
            "Nivel_Estres": 5, "Consultas_Medicas": 2, "Mes_Numero": 6}
FLOAT_COLS = ["IMC", "Glucosa_mg_dL", "Colesterol_mg_dL", "Horas_Sueno"]


def load_example():
    """Rellena el formulario con un registro real del conjunto de prueba (los faltantes usan el valor por defecto)."""
    rid = st.session_state.get("ejemplo")
    if rid in (None, "(ninguno)"):
        return
    row = test_df.loc[int(rid)]
    for col in FEATURES:
        val = row[col]
        if pd.isna(val):
            val = DEFAULTS[col]
        if col in CATEGORIES and col != "Mes_Numero":
            st.session_state[f"in_{col}"] = str(val)
        elif col == "Mes_Numero":
            st.session_state[f"in_{col}"] = int(val)
        else:
            st.session_state[f"in_{col}"] = float(val) if col in FLOAT_COLS else int(val)


def same_as_example(rec, row):
    """True si el formulario coincide con el registro de prueba cargado (faltantes = valor por defecto)."""
    for col in FEATURES:
        v = DEFAULTS[col] if pd.isna(row[col]) else row[col]
        same = str(v) == str(rec[col]) if (col in CATEGORIES and col != "Mes_Numero") else float(v) == float(rec[col])
        if not same:
            return False
    return True


with tab_pred:
    st.subheader("Consultar una predicción")
    st.info("Ingrese los datos de un registro o cargue uno del conjunto de prueba para compararlo con el notebook.")
    st.selectbox("Cargar un registro del conjunto de prueba (opcional)",
                 ["(ninguno)"] + [str(i) for i in test_df.index[~test_df[FEATURES].isna().any(axis=1)]],
                 key="ejemplo", on_change=load_example,
                 help="Solo se listan registros completos, para que la predicción coincida exactamente con la del notebook.")
    record = {}
    with st.form("formulario"):
        cols = st.columns(3)
        for k, col in enumerate(FEATURES):
            with cols[k % 3]:
                key = f"in_{col}"
                if col in CATEGORIES:
                    opts = list(CATEGORIES[col])
                    record[col] = st.selectbox(LABELS[col], opts, index=opts.index(DEFAULTS[col]),
                                               format_func=lambda v, c=col: CATEGORIES[c][v], key=key)
                else:
                    lo, hi = RANGES[col]
                    if col in FLOAT_COLS:
                        record[col] = st.number_input(LABELS[col], min_value=float(lo), max_value=float(hi),
                                                      value=float(DEFAULTS[col]), step=0.1, format="%.1f", key=key)
                    else:
                        record[col] = st.number_input(LABELS[col], min_value=int(lo), max_value=int(hi),
                                                      value=int(DEFAULTS[col]), step=1, key=key)
        submitted = st.form_submit_button("Predecir")

    if submitted:
        train_ranges = {c: tuple(v) for c, v in metrics["train_ranges"].items()}
        errors, warns = validate_record(record, train_ranges)
        if errors:
            for e in errors:
                st.error(e)
        else:
            for w in warns:
                st.warning(w)
            X_new = record_to_frame(record)
            pred = str(pipe.predict(X_new)[0])
            proba = pipe.predict_proba(X_new)[0]
            by_class = {c: float(proba[list(pipe.classes_).index(c)]) for c in CLASS_ORDER}
            c1, c2 = st.columns(2)
            c1.metric("Nivel de riesgo predicho", pred)
            c2.metric("Probabilidad de la clase predicha", f"{by_class[pred]:.1%}".replace(".", ","))
            fig, ax = plt.subplots(figsize=(5, 2.2))
            ax.barh(CLASS_ORDER[::-1], [by_class[c] for c in CLASS_ORDER[::-1]], color="steelblue")
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probabilidad")
            ax.set_title("Probabilidad por nivel de riesgo")
            st.pyplot(fig)
            rid = st.session_state.get("ejemplo")
            if rid not in (None, "(ninguno)"):
                row = test_df.loc[int(rid)]
                if same_as_example(record, row):
                    st.caption(f"Valor real del registro {rid} en el conjunto de prueba: **{row[TARGET]}**.")
            st.caption("Resultado de un prototipo académico con datos sintéticos: no es un diagnóstico y no "
                       "sustituye la decisión de un profesional de la salud.")

# ----------------------------------------------------------------------------- 4. Modelo
with tab_mod:
    st.subheader("Modelo final: regresión logística")
    st.markdown(
        "El pipeline guardado incluye imputación (mediana / categoría más frecuente), escalamiento, codificación "
        "one-hot y la regresión logística multiclase con `class_weight='balanced'`. Se entrenó con "
        f"{metrics['n_train']} registros y se evaluó con {metrics['n_test']} registros no vistos "
        "(división estratificada 75/25, `random_state=42`)."
    )
    t = metrics["test"]
    cols = st.columns(4)
    for c, (name, k) in zip(cols, [("Accuracy", "accuracy"), ("Precision macro", "precision_macro"),
                                    ("Recall macro", "recall_macro"), ("F1 macro", "f1_macro")]):
        c.metric(name + " (prueba)", f"{t[k]:.3f}".replace(".", ","))
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(4.4, 3.8))
        mat = metrics["confusion"]
        ax.imshow(mat, cmap="Blues")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, mat[i][j], ha="center", va="center", fontsize=15)
        ax.set_xticks(range(3), CLASS_ORDER)
        ax.set_yticks(range(3), CLASS_ORDER)
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Valor real")
        ax.set_title("Matriz de confusión (prueba)")
        st.pyplot(fig)
    with c2:
        st.markdown("**Resultados por clase (prueba)**")
        st.table(pd.DataFrame(metrics["per_class"]).T.rename(
            columns={"precision": "Precision", "recall": "Recall", "f1": "F1", "support": "Registros"}
        ).round(3))
        cv = metrics["cv_f1_macro"]
        st.markdown(f"**Validación cruzada (5 particiones), F1 macro:** {cv['mean']:.3f} ± {cv['std']:.3f}".replace(".", ","))
        errs_lo_hi = mat[0][2] + mat[2][0]
        st.markdown(f"Confusiones directas entre Bajo y Alto: **{errs_lo_hi}**. Los errores se dan entre niveles vecinos.")
    st.subheader("Limitaciones")
    st.markdown(
        "- **Datos sintéticos:** los resultados son educativos y no tienen validez clínica.\n"
        "- **Posible circularidad:** el nivel de riesgo pudo construirse con variables clínicas (como la glucosa) que "
        "también son predictoras; esto puede **inflar** el desempeño.\n"
        "- Evaluación con un solo conjunto de prueba pequeño; sin ajuste de hiperparámetros ni validación externa.\n"
        "- `Mes_Numero` se conserva como en el notebook, aunque su significado clínico es limitado.\n"
        "- El modelo describe asociaciones, no causas.\n"
        "- **Prototipo académico: no sustituye una decisión profesional.**"
    )
