"""Descarga y preparación de trayectorias históricas IBTrACS (NOAA/NCEI)."""
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from . import config

CUADRANTES = ("NE", "SE", "SW", "NW")
UMBRALES_KT = (34, 50, 64)
COLUMNAS_RADIOS = [f"USA_R{u}_{c}" for u in UMBRALES_KT for c in CUADRANTES]
COLUMNAS = ["SID", "SEASON", "BASIN", "NAME", "ISO_TIME", "LAT", "LON", "USA_WIND", "USA_ATCF_ID"] + COLUMNAS_RADIOS


def descargar(cuencas=config.CUENCAS_IBTRACS, carpeta=config.CARPETA_DATOS):
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    rutas = []
    for cuenca in cuencas:
        ruta = carpeta / f"ibtracs_{cuenca}.csv"
        if not ruta.exists():
            print(f"Descargando IBTrACS {cuenca}...")
            with requests.get(config.URL_IBTRACS.format(cuenca=cuenca), stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(ruta, "wb") as f:
                    for bloque in r.iter_content(1 << 20):
                        f.write(bloque)
        rutas.append(ruta)
    return rutas


def cargar(rutas, anio_minimo=config.ANIO_MINIMO_HISTORICO):
    """Lee los CSV de IBTrACS y deja una fila por (tormenta, hora) con radios en km."""
    marcos = []
    for ruta in rutas:
        # La segunda línea del CSV trae unidades, no datos.
        df = pd.read_csv(ruta, usecols=COLUMNAS, skiprows=[1], na_values=[" ", ""], keep_default_na=True, low_memory=False)
        marcos.append(df)
    df = pd.concat(marcos, ignore_index=True)
    df = df[df["SEASON"] >= anio_minimo].copy()
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"])
    for col in ["LAT", "LON", "USA_WIND"] + COLUMNAS_RADIOS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return normalizar(df)


def normalizar(df):
    """Renombra columnas al esquema interno y convierte radios de millas náuticas a km."""
    salida = pd.DataFrame({
        "sid": df["SID"].values,
        "temporada": df["SEASON"].values,
        "nombre": df["NAME"].values,
        "tiempo": pd.to_datetime(df["ISO_TIME"]).values,
        "lat": df["LAT"].values,
        "lon": df["LON"].values,
        "vmax_kt": df["USA_WIND"].values,
        "atcf": df["USA_ATCF_ID"].values if "USA_ATCF_ID" in df else None,
    })
    for u in UMBRALES_KT:
        for c in CUADRANTES:
            salida[f"r{u}_{c.lower()}_km"] = df[f"USA_R{u}_{c}"].values * 1.852
    return salida.dropna(subset=["lat", "lon"]).sort_values(["sid", "tiempo"]).reset_index(drop=True)


def interpolar_horario(trayectoria):
    """Interpola una tormenta a pasos de 1 hora (posición, viento y radios).

    Los radios faltantes se dejan en 0: si la agencia no reportó radio de 34 kt,
    se asume que en ese cuadrante no había vientos de esa fuerza.
    """
    t = trayectoria.drop(columns=["atcf"], errors="ignore").drop_duplicates("tiempo").set_index("tiempo").sort_index()
    numericas = t.select_dtypes(include=[np.number]).drop(columns=["temporada"], errors="ignore")
    horas = pd.date_range(t.index.min(), t.index.max(), freq="1h")
    interp = numericas.reindex(numericas.index.union(horas)).interpolate(method="time", limit_area="inside").loc[horas]
    columnas_radio = [c for c in interp.columns if c.startswith("r") and c.endswith("_km")]
    interp[columnas_radio] = interp[columnas_radio].fillna(0)
    interp["vmax_kt"] = interp["vmax_kt"].ffill().bfill()
    interp = interp.reset_index(names="tiempo")
    for col in ("sid", "temporada", "nombre"):
        interp[col] = trayectoria[col].iloc[0]
    return interp


def tormentas_relevantes(df, caja=config.CAJA_MEXICO, margen_grados=3.0):
    """SIDs de tormentas que pasaron cerca de México."""
    dentro = (
        df["lat"].between(caja["lat_min"] - margen_grados, caja["lat_max"] + margen_grados)
        & df["lon"].between(caja["lon_min"] - margen_grados, caja["lon_max"] + margen_grados)
    )
    return sorted(df.loc[dentro, "sid"].unique())


def buscar_tormenta(df, nombre, temporada):
    sel = df[(df["nombre"].str.upper() == nombre.upper()) & (df["temporada"] == int(temporada))]
    if sel.empty:
        raise ValueError(f"No encontré {nombre} {temporada} en IBTrACS")
    return sel["sid"].iloc[0]


def atcf_de_tormentas(df, sids):
    """Identificadores ATCF del NHC (p. ej. EP182023) de cada tormenta; algunas cruzan de cuenca y tienen dos."""
    sel = df[df["sid"].isin(sids)].dropna(subset=["atcf"])
    sel = sel[sel["atcf"].str[:2].isin(["AL", "EP"])]
    return sel.groupby("sid")["atcf"].unique().map(list).to_dict()
