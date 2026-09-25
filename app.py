import joblib
import numpy as np
import pandas as pd
import streamlit as st

RUTA_MODELO = "model.pkl"
RUTA_MUESTRA = "datos_muestra.csv"
SIN_DATO = "(sin dato)"
COLUMNAS_VISTA = [
    "TransactionID", "TransactionAmt", "ProductCD", "card4", "card6",
    "isFraud", "prob_fraude", "alerta",
]


@st.cache_resource
def cargar_modelo():
    return joblib.load(RUTA_MODELO)


@st.cache_data
def cargar_muestra():
    return pd.read_csv(RUTA_MUESTRA)


def predecir(artefacto, transacciones):
    return artefacto["pipeline"].predict_proba(transacciones[artefacto["columnas_entrada"]])[:, 1]


def selector_categoria(contenedor, etiqueta, valores_posibles, valor_actual):
    opciones = sorted(valores_posibles.dropna().unique().tolist()) + [SIN_DATO]
    actual = SIN_DATO if pd.isna(valor_actual) else valor_actual
    elegido = contenedor.selectbox(etiqueta, opciones, index=opciones.index(actual))
    return np.nan if elegido == SIN_DATO else elegido


def mismo_valor(a, b):
    return (pd.isna(a) and pd.isna(b)) or a == b


def describir_transaccion(fila):
    etiqueta = "FRAUDE" if fila["isFraud"] == 1 else "Legítima"
    return f"{etiqueta} · ID {fila['TransactionID']} · USD {fila['TransactionAmt']:,.2f} · producto {fila['ProductCD']}"


def mostrar_veredicto(alerta, es_fraude):
    if alerta and es_fraude:
        st.success("Acierto: el modelo detectó un fraude real.")
    elif not alerta and not es_fraude:
        st.success("Acierto: el modelo aprobó una transacción legítima.")
    elif alerta:
        st.warning("Falsa alarma: el modelo marcó como sospechosa una transacción legítima.")
    else:
        st.error("Fraude no detectado: el modelo aprobó una transacción fraudulenta.")


st.set_page_config(page_title="Detección de fraude", layout="centered")

artefacto = cargar_modelo()
muestra = cargar_muestra()
umbral = artefacto["umbral"]

st.title("Detección de fraude en transacciones")
st.caption(
    f"Pipeline de scikit-learn + LightGBM entrenado sobre IEEE-CIS · "
    f"se genera una alerta cuando la probabilidad supera {umbral:.1%}"
)

pestania_individual, pestania_lote = st.tabs(["Transacción individual", "Lote desde CSV"])

with pestania_individual:
    st.write("Elegí una transacción real del período de test y, si querés, modificá algunos campos.")

    filtro = st.radio("Mostrar", ["Todas", "Solo fraudes", "Solo legítimas"], horizontal=True)
    if filtro == "Solo fraudes":
        opciones = muestra.index[muestra["isFraud"] == 1]
    elif filtro == "Solo legítimas":
        opciones = muestra.index[muestra["isFraud"] == 0]
    else:
        opciones = muestra.index

    indice = st.selectbox("Transacción", opciones, format_func=lambda i: describir_transaccion(muestra.loc[i]))
    transaccion = muestra.loc[[indice]].copy()
    original = transaccion.iloc[0]

    transaccion["TransactionAmt"] = st.number_input(
        "Monto (USD)", min_value=0.0, value=float(original["TransactionAmt"]), step=10.0
    )
    transaccion["ProductCD"] = selector_categoria(st, "Producto", muestra["ProductCD"], original["ProductCD"])
    transaccion["card4"] = selector_categoria(st, "Red de la tarjeta", muestra["card4"], original["card4"])
    transaccion["card6"] = selector_categoria(st, "Tipo de tarjeta", muestra["card6"], original["card6"])

    modificada = not all(
        mismo_valor(transaccion[col].iloc[0], original[col])
        for col in ["TransactionAmt", "ProductCD", "card4", "card6"]
    )
    probabilidad = predecir(artefacto, transaccion)[0]
    alerta = probabilidad >= umbral
    es_fraude = original["isFraud"] == 1

    st.divider()
    st.metric("Probabilidad de fraude", f"{probabilidad:.1%}")
    st.metric("Decisión del modelo", "ALERTA: enviar a revisión" if alerta else "Aprobada")
    st.metric("Etiqueta real", "Fraude" if es_fraude else "Legítima")

    if modificada:
        st.info("Modificaste campos: la etiqueta real corresponde a la transacción original y ya no sirve para evaluar el acierto.")
    else:
        mostrar_veredicto(alerta, es_fraude)

with pestania_lote:
    archivo = st.file_uploader("CSV con transacciones (mismas columnas que datos_muestra.csv)", type="csv")
    if archivo is None:
        st.info("Sin archivo cargado: se usa datos_muestra.csv (20 fraudes y 80 transacciones legítimas).")
        lote = muestra
    else:
        lote = pd.read_csv(archivo)

    faltantes = [c for c in artefacto["columnas_entrada"] if c not in lote.columns]
    if faltantes:
        st.error(f"Al CSV le faltan {len(faltantes)} columnas, por ejemplo: {', '.join(faltantes[:5])}")
        st.stop()

    resultado = lote.copy()
    resultado["prob_fraude"] = predecir(artefacto, lote).round(3)
    resultado["alerta"] = resultado["prob_fraude"] >= umbral

    st.metric("Transacciones", len(resultado))
    st.metric("Alertas", int(resultado["alerta"].sum()))
    if "isFraud" in resultado.columns:
        detectados = int((resultado["alerta"] & (resultado["isFraud"] == 1)).sum())
        st.metric("Fraudes reales detectados", f"{detectados} de {int(resultado['isFraud'].sum())}")

    vista = resultado[[c for c in COLUMNAS_VISTA if c in resultado.columns]]
    vista = vista.sort_values("prob_fraude", ascending=False)
    st.dataframe(vista, hide_index=True)
    st.download_button("Descargar resultados", vista.to_csv(index=False), "predicciones.csv", "text/csv")
