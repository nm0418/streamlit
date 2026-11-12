# 🩺 Clasificación del nivel de riesgo en salud (datos sintéticos)

Prototipo académico del curso **Minería de Datos** (Corporación Universitaria de la Costa, Ingeniería de Sistemas). Aplicación en Streamlit que despliega el mejor clasificador de la actividad de modelado (`Modeling_Activity_NelsyMadera.ipynb`): **regresión logística** para predecir `Nivel_Riesgo` (Bajo / Medio / Alto).

> ⚠️ Los datos son **sintéticos** y la aplicación es un prototipo académico. No tiene validez clínica y **no sustituye la decisión de un profesional de la salud**.

- **Aplicación pública:** _(pegar aquí la URL de Streamlit Community Cloud)_
- **Integrantes:** Nelsy Madera _(agregar los nombres completos de los demás integrantes)_

## Datos
`Dataset_Sintetico_Salud_.xlsx` (hoja `Datos`): 500 registros y 14 columnas; 3 duplicados exactos eliminados → 497 registros. Faltantes en IMC (5), glucosa (6), colesterol (3), actividad física (4) y horas de sueño (7), imputados dentro del pipeline. `ID_Paciente` no se usa como predictor.

## Modelo
Pipeline guardado con `joblib` (`models/pipeline_salud.joblib`): imputación (mediana / categoría más frecuente) + escalamiento + one-hot + `LogisticRegression(max_iter=2000, class_weight="balanced")`. División estratificada 75/25 (`random_state=42`): 372 registros de entrenamiento y 125 de prueba. La app **carga** el archivo; no entrena en cada ejecución.

Resultados del notebook (prueba): accuracy 0,856 · precision macro 0,858 · recall macro 0,862 · F1 macro 0,859 · 107 de 125 aciertos · recall por clase: Bajo 0,860, Medio 0,816, Alto 0,909 · sin confusiones directas Bajo–Alto. Validación cruzada (5 particiones), F1 macro: 0,837 ± 0,044. Comparación: el árbol de decisión obtuvo accuracy 0,680 y F1 macro 0,640 en validación cruzada, por lo que se despliega la regresión logística.

`train_model.py` reproduce estos valores y los compara con los del notebook al terminar.

## Estructura
```
app.py                  # aplicación Streamlit
health_utils.py         # variables, etiquetas, rangos y validación
train_model.py          # reproduce el modelo del notebook y guarda artefactos
test_consistency.py     # pruebas: caso normal, entradas inválidas, coincidencia con el notebook
data/Dataset_Sintetico_Salud_.xlsx
data/datos_limpios.csv, data/test_set.csv        (los genera train_model.py)
models/pipeline_salud.joblib, models/metrics.json (los genera train_model.py)
Modeling_Activity_NelsyMadera.ipynb   # notebook
requirements.txt
```

## Cómo ejecutarlo
```bash
pip install -r requirements.txt
# 1) copiar Dataset_Sintetico_Salud_.xlsx en data/
python train_model.py          # genera el pipeline y compara con el notebook
python test_consistency.py     # pruebas sin interfaz
streamlit run app.py
```

## Validación de entradas
Rangos: edad 18–100, IMC 10–60, glucosa 40–400 mg/dL, presión sistólica 70–250 mmHg, colesterol 80–500 mg/dL, sueño 0–16 h, estrés 0–10, consultas 0–30, mes 1–12. Las variables enteras deben ser enteras y las categorías deben ser válidas. Un valor válido pero fuera del rango observado en el entrenamiento genera una advertencia.

## Limitaciones
- Datos sintéticos; resultados solo educativos.
- **Posible circularidad:** `Nivel_Riesgo` pudo construirse con variables clínicas (como la glucosa) que también son predictoras; el desempeño puede estar inflado.
- Un solo conjunto de prueba pequeño, sin ajuste de hiperparámetros ni validación externa.
- `Mes_Numero` se conserva como en el notebook, aunque su significado clínico es limitado.
- Describe asociaciones, no causas.

## Declaración y evaluación del uso de inteligencia artificial
| Elemento | Información |
|---|---|
| Herramienta utilizada | Claude (Anthropic) |
| Propósito | Apoyar la preparación del repositorio: `app.py`, `health_utils.py`, `train_model.py`, `test_consistency.py` y este README. |
| Prompt principal | _(pegar aquí el prompt completo enviado, tal como lo escribió el grupo)_ |
| Cómo se verificó la respuesta | _(Completar: ejecución de `train_model.py` con el Excel real y comprobación de que coincide con el notebook; `test_consistency.py`; prueba de la app con `streamlit run app.py` y comparación de un registro de prueba con el notebook.)_ |
| Problemas identificados | En la primera versión el asistente no tuvo acceso al Excel; luego `train_model.py` y `test_consistency.py` se ejecutaron con el Excel real y reprodujeron las métricas del notebook (accuracy 0,856; F1 macro 0,859; CV 0,837 ± 0,044). Pendiente: la interfaz de `app.py` no se ha ejecutado con Streamlit real ni publicado en Streamlit Cloud. Además, 4 registros de prueba tienen faltantes; se excluyeron del selector de ejemplos porque la app no imputa y no coincidirían con el notebook. _(Agregar otros problemas que encuentre el grupo.)_ |
| Ajustes realizados | _(Completar: qué corrigió, eliminó o modificó el grupo.)_ |
| Decisión final | _(El grupo debe indicar qué conservó y por qué.)_ |

Todos los integrantes deben comprender el código entregado.
