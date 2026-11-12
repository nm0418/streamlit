# 🩺 Clasificación del nivel de riesgo en salud (datos sintéticos)

Prototipo académico del curso **Minería de Datos** (Corporación Universitaria de la Costa, Ingeniería de Sistemas). Aplicación en Streamlit que despliega el mejor clasificador de la actividad de modelado (`Modeling_Activity_NelsyMadera.ipynb`): **regresión logística** para predecir `Nivel_Riesgo` (Bajo / Medio / Alto).

> ⚠️ Los datos son **sintéticos** y la aplicación es un prototipo académico. No tiene validez clínica y **no sustituye la decisión de un profesional de la salud**.

- **Aplicación pública:** https://app-hrykbxa3fz8qnmv2ipb4qt.streamlit.app/
- **Integrantes:** Nelsy Madera

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
| Prompt principal | Paso a paso explicado para subir y crear una app con estos datos en streamlit|
| Cómo se verificó la respuesta | Se ejecutó train_model.py con el Excel real y las métricas coincidieron con las del notebook (accuracy 0,856; F1 macro 0,859; validación cruzada 0,837 ± 0,044). Se ejecutó test_consistency.py (caso normal, 7 entradas inválidas y un registro de prueba). En la app desplegada se probó un caso normal (Bajo, 56,3 %), el registro 314 del conjunto de prueba (Medio, 86,5 %, igual que el notebook) y una edad de 90 años, que generó la advertencia de valor fuera del rango de entrenamiento. [Agrega aquí qué pasó con la prueba de edad 150 o similar.]
| Problemas identificados | La primera versión del código se escribió sin acceso al Excel y no se había probado con los datos reales ni en Streamlit. Los registros de prueba con valores faltantes no coincidían con el notebook, porque la app usa valores por defecto en vez de la imputación del pipeline. Al subir el proyecto, los archivos quedaron anidados y el Excel en un lugar incorrecto.
| Ajustes realizados | Se excluyeron del selector los registros de prueba con faltantes. Se fijó scikit-learn==1.9.1 en requirements.txt. Se reorganizaron los archivos en el repositorio |
| Decisión final | Se conservó la regresión logística, que superó al árbol de decisión en la prueba y en la validación cruzada. No se modificó el modelo del notebook. Se mantiene la advertencia de que el desempeño puede estar inflado si Nivel_Riesgo se construyó con variables clínicas. |

