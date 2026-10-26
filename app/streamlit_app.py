"""Visor de alertas: streamlit run app/streamlit_app.py"""
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

SALIDAS = Path(__file__).resolve().parent.parent / "salidas"

st.set_page_config(page_title="Riesgo de huracanes", layout="wide")
st.title("Riesgo de huracanes por ubicación")

archivos = sorted(SALIDAS.glob("*_alertas.csv"), reverse=True)
if not archivos:
    st.info("Aún no hay alertas. Corre `python -m huracan_ia simular ...` o `python -m huracan_ia alertar --activas`.")
    st.stop()

elegido = st.selectbox("Emisión", archivos, format_func=lambda p: p.name.replace("_alertas.csv", ""))
base = str(elegido).replace("_alertas.csv", "")
alertas = pd.read_csv(elegido)

c1, c2, c3 = st.columns(3)
for col, nivel in zip((c1, c2, c3), ("Alto", "Medio", "Bajo")):
    col.metric(f"Nivel {nivel} (modelo)", int((alertas["nivel"] == nivel).sum()))

izq, der = st.columns([3, 2])
with izq:
    mapa = Path(base + "_mapa.html")
    if mapa.exists():
        components.html(mapa.read_text(encoding="utf-8"), height=560)
with der:
    boletin = Path(base + "_boletin.md")
    if boletin.exists():
        st.markdown(boletin.read_text(encoding="utf-8"))

st.subheader("Modelo vs. índice de pesos fijos")
st.dataframe(pd.crosstab(alertas["nivel"], alertas["nivel_base"]), use_container_width=True)
st.dataframe(alertas, use_container_width=True, hide_index=True)
