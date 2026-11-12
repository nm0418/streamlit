"""Reproduce el modelo de clasificación del notebook (Modeling_Activity_NelsyMadera.ipynb), lo guarda con joblib
y escribe métricas, datos limpios y conjunto de prueba.

Requisito: colocar Dataset_Sintetico_Salud_.xlsx (hoja "Datos") en la carpeta data/.
Uso:  python train_model.py
La app NO entrena: solo carga models/pipeline_salud.joblib.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from health_utils import CAT_COLS, CLASS_ORDER, FEATURES, NUM_COLS, SEED, TARGET

ROOT = Path(__file__).parent
XLSX_NAMES = ["Dataset_Sintetico_Salud_.xlsx", "Dataset_Sintetico_Salud_EDA.xlsx"]
CLEAN_PATH = ROOT / "data" / "datos_limpios.csv"
TEST_PATH = ROOT / "data" / "test_set.csv"
MODEL_PATH = ROOT / "models" / "pipeline_salud.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"

# Valores del notebook, para comprobar que este script lo reproduce (tolerancia por redondeo)
NOTEBOOK = {"accuracy": 0.856, "precision_macro": 0.858, "recall_macro": 0.862, "f1_macro": 0.859,
            "cv_f1_mean": 0.837, "cv_f1_std": 0.044, "n_rows": 497}


def find_xlsx() -> Path:
    for name in XLSX_NAMES:
        p = ROOT / "data" / name
        if p.exists():
            return p
    raise FileNotFoundError("No se encontró data/Dataset_Sintetico_Salud_.xlsx. Cópialo en la carpeta data/.")


def load_clean() -> pd.DataFrame:
    df = pd.read_excel(find_xlsx(), sheet_name="Datos")
    df["Mes_Registro"] = pd.to_datetime(df["Mes_Registro"])
    df = df.drop_duplicates().copy()                      # igual que el notebook
    df["Mes_Numero"] = df["Mes_Registro"].dt.month
    return df


def build_pipeline() -> Pipeline:
    prep = ColumnTransformer([
        ("num", Pipeline([("imputation", SimpleImputer(strategy="median")),
                          ("scaling", StandardScaler())]), NUM_COLS),
        ("cat", Pipeline([("imputation", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CAT_COLS),
    ])
    return Pipeline([("preprocessing", prep),
                     ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED))])


def build_and_save():
    df = load_clean()
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)

    pipe = build_pipeline().fit(X_train, y_train)
    pred = pipe.predict(X_test)
    cv = cross_val_score(build_pipeline(), X, y, cv=5, scoring="f1_macro")   # igual que el notebook

    rep = classification_report(y_test, pred, labels=CLASS_ORDER, output_dict=True, zero_division=0)
    metrics = {
        "test": {"accuracy": accuracy_score(y_test, pred),
                 "precision_macro": precision_score(y_test, pred, average="macro", zero_division=0),
                 "recall_macro": recall_score(y_test, pred, average="macro", zero_division=0),
                 "f1_macro": f1_score(y_test, pred, average="macro", zero_division=0)},
        "per_class": {c: {"precision": rep[c]["precision"], "recall": rep[c]["recall"],
                          "f1": rep[c]["f1-score"], "support": int(rep[c]["support"])} for c in CLASS_ORDER},
        "confusion": confusion_matrix(y_test, pred, labels=CLASS_ORDER).tolist(),
        "cv_f1_macro": {"mean": float(cv.mean()), "std": float(cv.std())},
        "n_total": int(len(df)), "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "class_counts": {c: int((y == c).sum()) for c in CLASS_ORDER},
        "train_ranges": {c: [float(X_train[c].min()), float(X_train[c].max())] for c in NUM_COLS},
    }
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    df.drop(columns=["Mes_Registro"]).to_csv(CLEAN_PATH, index_label="id_original")
    out = X_test.copy()
    out[TARGET] = y_test
    out.to_csv(TEST_PATH, index_label="id_original")
    return pipe, metrics


def compare_with_notebook(m):
    got = {**m["test"], "cv_f1_mean": m["cv_f1_macro"]["mean"], "cv_f1_std": m["cv_f1_macro"]["std"],
           "n_rows": m["n_total"]}
    ok = True
    for k, expected in NOTEBOOK.items():
        diff = abs(round(got[k], 3) - expected)
        flag = "OK " if diff < 0.0015 else "DIFERENTE"
        ok &= diff < 0.0015
        print(f"  {flag} {k}: script = {got[k]:.3f} | notebook = {expected}")
    print("Coincide con el notebook." if ok else
          "ATENCIÓN: hay diferencias con el notebook. Revisa que el Excel sea el mismo y la versión de scikit-learn.")
    return ok


if __name__ == "__main__":
    _, metrics = build_and_save()
    compare_with_notebook(metrics)
