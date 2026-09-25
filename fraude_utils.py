"""Funciones del pipeline de preprocesamiento.

Viven en un módulo aparte (no en el notebook) para que el modelo guardado con
joblib pueda cargarse desde cualquier script, como la app de Streamlit.
"""

import numpy as np
import pandas as pd

SEGUNDOS_POR_HORA = 3600
HORAS_POR_DIA = 24


def a_texto(X: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(X).astype(str)


def extraer_hora(X: pd.DataFrame) -> pd.DataFrame:
    segundos = np.asarray(X, dtype="float64")[:, 0]
    indice = X.index if isinstance(X, pd.DataFrame) else None
    return pd.DataFrame(
        {"hora_del_dia": (segundos // SEGUNDOS_POR_HORA) % HORAS_POR_DIA}, index=indice
    )


def nombre_hora(transformer, input_features):
    return ["hora_del_dia"]
