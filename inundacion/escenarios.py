"""Escenarios de lluvia para un dominio: observada (CMORPH), trayectoria real y pronóstico del NHC.

Los tres producen una serie horaria de lluvia sobre el centro del dominio, sobre la
misma ventana de tiempo, para que las inundaciones simuladas se puedan comparar.
"""
import re
import zipfile

import numpy as np
import pandas as pd

from . import config, lluvia, modelo_lluvia

ZONAS_NHC = {"CST": -6, "CDT": -5, "MST": -7, "MDT": -6, "PST": -8, "PDT": -7, "EST": -5, "EDT": -4, "AST": -4, "HST": -10}


def centro(dominio):
    return (dominio["lat_min"] + dominio["lat_max"]) / 2, (dominio["lon_min"] + dominio["lon_max"]) / 2


def ventana(inicio, fin):
    return pd.date_range(pd.Timestamp(inicio).floor("h"), pd.Timestamp(fin).floor("h"), freq="1h")


def observada(dominio, horas):
    lat, lon = centro(dominio)
    return lluvia.serie_en_punto(lat, lon, horas[0], horas[-1]).reindex(horas).fillna(0.0)


def con_trayectoria(mdl, trayectoria, dominio, horas, emision, metodo="lightgbm"):
    """Lluvia del modelo a lo largo de una trayectoria, desde la emisión; cero fuera de ese periodo."""
    lat, lon = centro(dominio)
    trayectoria = trayectoria[trayectoria["tiempo"] >= pd.Timestamp(emision).floor("6h")]
    serie = modelo_lluvia.serie_en_punto(mdl, trayectoria, lat, lon, emision, metodo=metodo)
    return serie.reindex(horas).fillna(0.0)


def emision_nhc(texto):
    """'1200 PM CST Mon Sep 23 2024' -> Timestamp en UTC."""
    m = re.match(r"(\d{3,4}) (AM|PM) ([A-Z]{3}) \w{3} (\w{3}) (\d{1,2}) (\d{4})", texto.strip())
    if not m:
        raise ValueError(f"No reconozco la fecha de emisión del NHC: {texto!r}")
    hhmm, ampm, zona, mes, dia, anio = m.groups()
    hora, minuto = int(hhmm[:-2]) % 12 + (12 if ampm == "PM" else 0), int(hhmm[-2:])
    local = pd.Timestamp(f"{dia} {mes} {anio} {hora:02d}:{minuto:02d}")
    return local - pd.Timedelta(hours=ZONAS_NHC[zona])


def pronostico_nhc(ruta_zip):
    """Lee el zip '5day' del NHC y devuelve (emisión UTC, trayectoria con tiempo, lat, lon, vmax_kt)."""
    import geopandas as gpd

    with zipfile.ZipFile(ruta_zip) as z:
        capa = next(n for n in z.namelist() if n.lower().endswith("_pts.shp"))[:-4]
    pts = gpd.read_file(ruta_zip, layer=capa)
    emision = emision_nhc(str(pts["ADVDATE"].iloc[0]))
    # TAU cuenta desde la hora sinóptica previa a la emisión (emisión 3 h después de la hora sinóptica).
    base = emision.floor("6h") if emision.hour % 6 else emision
    tray = pd.DataFrame({
        "tiempo": base + pd.to_timedelta(pd.to_numeric(pts["TAU"]), unit="h"),
        "lat": pd.to_numeric(pts["LAT"]), "lon": pd.to_numeric(pts["LON"]),
        "vmax_kt": pd.to_numeric(pts["MAXWIND"]),
    }).sort_values("tiempo").reset_index(drop=True)
    tray["nombre"] = str(pts["STORMNAME"].iloc[0]) if "STORMNAME" in pts else "NHC"
    return emision, tray


def lluvia_total_mm(serie):
    return float(np.nansum(serie))
