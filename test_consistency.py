"""Pruebas sin abrir Streamlit:  python test_consistency.py   (requiere haber ejecutado train_model.py)

1) caso normal, 2) entradas inválidas, 3) registro conocido del conjunto de prueba: la predicción del archivo
guardado debe coincidir con la del notebook (mismo procedimiento reentrenado desde el Excel).
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from health_utils import FEATURES, SEED, TARGET, record_to_frame, validate_record
from train_model import METRICS_PATH, MODEL_PATH, build_pipeline, load_clean

pipe = joblib.load(MODEL_PATH)
metrics = json.loads(METRICS_PATH.read_text())
train_ranges = {c: tuple(v) for c, v in metrics["train_ranges"].items()}

normal = {"Edad": 45, "Genero": "Femenino", "IMC": 26.0, "Glucosa_mg_dL": 100.0, "Presion_Sistolica_mmHg": 120,
          "Colesterol_mg_dL": 190.0, "Actividad_Fisica": "Moderada", "Fumador": "No", "Horas_Sueno": 7.0,
          "Nivel_Estres": 5, "Consultas_Medicas": 2, "Mes_Numero": 6}
errs, _ = validate_record(normal, train_ranges)
assert not errs, errs
X1 = record_to_frame(normal)
print(f"[1] Caso normal -> {pipe.predict(X1)[0]} | probabilidades: "
      + ", ".join(f"{c}={p:.3f}" for c, p in zip(pipe.classes_, pipe.predict_proba(X1)[0])))

invalidos = {
    "edad negativa": {**normal, "Edad": -5},
    "glucosa fuera de rango": {**normal, "Glucosa_mg_dL": 900.0},
    "texto en campo numérico": {**normal, "IMC": "alto"},
    "categoría inexistente": {**normal, "Actividad_Fisica": "Extrema"},
    "falta un campo": {k: v for k, v in normal.items() if k != "Horas_Sueno"},
    "decimal en campo entero": {**normal, "Nivel_Estres": 5.5},
    "mes inexistente": {**normal, "Mes_Numero": 13},
}
for nombre, rec in invalidos.items():
    errs, _ = validate_record(rec, train_ranges)
    assert errs, f"No se detectó el error: {nombre}"
    print(f"[2] Entrada inválida ({nombre}) -> rechazada: {errs[0]}")
_, warns = validate_record({**normal, "Edad": 99}, train_ranges)
print("[2] Valor válido fuera del rango de entrenamiento ->", "advertencia generada" if warns else "sin advertencia (dentro del rango)")

df = load_clean()
X_train, X_test, y_train, y_test = train_test_split(df[FEATURES], df[TARGET], test_size=0.25,
                                                    random_state=SEED, stratify=df[TARGET])
ref = build_pipeline().fit(X_train, y_train)
assert np.array_equal(ref.predict(X_test), pipe.predict(X_test))
assert np.allclose(ref.predict_proba(X_test), pipe.predict_proba(X_test))
idx = X_test.index[0]
rec = {c: (X_test.loc[idx, c].item() if hasattr(X_test.loc[idx, c], "item") else X_test.loc[idx, c]) for c in FEATURES}
rec = {c: (None if (isinstance(v, float) and np.isnan(v)) else v) for c, v in rec.items()}
pa = pipe.predict_proba(X_test.loc[[idx]])[0]
pn = ref.predict_proba(X_test.loc[[idx]])[0]
assert np.allclose(pa, pn)
print(f"[3] Registro {idx} del conjunto de prueba: app = {np.round(pa, 6).tolist()} | notebook = {np.round(pn, 6).tolist()} "
      f"(real: {df.loc[idx, TARGET]}) -> coinciden")
print("Todas las pruebas pasaron.")
