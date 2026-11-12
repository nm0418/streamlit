"""Definiciones compartidas: variables, etiquetas, rangos y validación (dataset sintético de salud)."""
import pandas as pd

NUM_COLS = ["Edad", "IMC", "Glucosa_mg_dL", "Presion_Sistolica_mmHg", "Colesterol_mg_dL",
            "Horas_Sueno", "Nivel_Estres", "Consultas_Medicas", "Mes_Numero"]
CAT_COLS = ["Genero", "Actividad_Fisica", "Fumador"]
FEATURES = ["Edad", "Genero", "IMC", "Glucosa_mg_dL", "Presion_Sistolica_mmHg", "Colesterol_mg_dL",
            "Actividad_Fisica", "Fumador", "Horas_Sueno", "Nivel_Estres", "Consultas_Medicas", "Mes_Numero"]
TARGET = "Nivel_Riesgo"
CLASS_ORDER = ["Bajo", "Medio", "Alto"]
SEED = 42  # SEMILLA del notebook

LABELS = {
    "Edad": "Edad (años)",
    "Genero": "Género",
    "IMC": "Índice de masa corporal (IMC)",
    "Glucosa_mg_dL": "Glucosa (mg/dL)",
    "Presion_Sistolica_mmHg": "Presión sistólica (mmHg)",
    "Colesterol_mg_dL": "Colesterol (mg/dL)",
    "Actividad_Fisica": "Actividad física",
    "Fumador": "¿Fuma?",
    "Horas_Sueno": "Horas de sueño por día",
    "Nivel_Estres": "Nivel de estrés (0 a 10)",
    "Consultas_Medicas": "Consultas médicas (número)",
    "Mes_Numero": "Mes del registro",
}

CATEGORIES = {  # valor del dataset -> texto mostrado
    "Genero": {"Femenino": "Femenino", "Masculino": "Masculino", "Otro/No reporta": "Otro / No reporta"},
    "Actividad_Fisica": {"Baja": "Baja", "Moderada": "Moderada", "Alta": "Alta"},
    "Fumador": {"No": "No", "Sí": "Sí"},
    "Mes_Numero": {i: n for i, n in enumerate(
        ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
         "septiembre", "octubre", "noviembre", "diciembre"], start=1)},
}

# Límites de validación (razonables para este tipo de variable; se advierte aparte si salen del rango entrenado)
RANGES = {
    "Edad": (18, 100),
    "IMC": (10.0, 60.0),
    "Glucosa_mg_dL": (40.0, 400.0),
    "Presion_Sistolica_mmHg": (70, 250),
    "Colesterol_mg_dL": (80.0, 500.0),
    "Horas_Sueno": (0.0, 16.0),
    "Nivel_Estres": (0, 10),
    "Consultas_Medicas": (0, 30),
}
INTEGER_COLS = ["Edad", "Presion_Sistolica_mmHg", "Nivel_Estres", "Consultas_Medicas"]
MES_KEYS = list(CATEGORIES["Mes_Numero"])


def validate_record(record: dict, train_ranges: dict | None = None):
    """Devuelve (errores, advertencias). Los errores impiden predecir."""
    errors, warnings = [], []
    for col in FEATURES:
        val = record.get(col)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            errors.append(f"Falta el valor de '{LABELS[col]}'.")
            continue
        if col in CATEGORIES:
            if val not in CATEGORIES[col]:
                errors.append(f"'{LABELS[col]}': categoría no válida ({val!r}). Opciones: {list(CATEGORIES[col])}.")
            continue
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            errors.append(f"'{LABELS[col]}' debe ser numérico (recibido: {val!r}).")
            continue
        if col in INTEGER_COLS and int(val) != val:
            errors.append(f"'{LABELS[col]}' debe ser un número entero.")
            continue
        lo, hi = RANGES[col]
        if not (lo <= val <= hi):
            errors.append(f"'{LABELS[col]}' debe estar entre {lo} y {hi} (recibido: {val}).")
            continue
        if train_ranges and col in train_ranges:
            tlo, thi = train_ranges[col]
            if val < tlo or val > thi:
                warnings.append(f"'{LABELS[col]}' = {val} está fuera del rango observado en el entrenamiento "
                                f"({tlo:g} a {thi:g}); la predicción puede ser menos confiable.")
    return errors, warnings


def record_to_frame(record: dict) -> pd.DataFrame:
    return pd.DataFrame([{c: record[c] for c in FEATURES}], columns=FEATURES)
