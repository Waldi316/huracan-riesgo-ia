"""Pronósticos de trayectoria en un formato común.

Un pronóstico es un DataFrame con columnas: horas, lat, lon, vmax_kt
(horas = 0 es la posición actual). Puede venir de dos lados:

- Del NHC (operación): el shapefile de puntos del pronóstico a 5 días.
- De la persistencia (entrenamiento y backtest): se extrapola el movimiento
  de las últimas 12 horas, que es lo único que se sabía en ese momento sin
  archivo de pronósticos oficiales.
"""
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from . import config, geo


def por_persistencia(trayectoria_horaria, t0, horizonte=config.HORIZONTE_HORAS,
                     paso=config.PASO_PRONOSTICO_HORAS, horas_atras=12):
    """Extrapola el movimiento reciente de la tormenta. Devuelve (estado, pronostico) o None."""
    t = trayectoria_horaria.set_index("tiempo")
    t_previo = t0 - pd.Timedelta(hours=horas_atras)
    if t0 not in t.index or t_previo not in t.index:
        return None
    actual, previo = t.loc[t0], t.loc[t_previo]
    dist = geo.distancia_km(previo["lat"], previo["lon"], actual["lat"], actual["lon"])
    rumbo = geo.rumbo_deg(previo["lat"], previo["lon"], actual["lat"], actual["lon"])
    velocidad = dist / horas_atras
    horas = np.arange(0, horizonte + 1, paso)
    lat, lon = geo.destino(actual["lat"], actual["lon"], rumbo, velocidad * horas)
    pronostico = pd.DataFrame({"horas": horas, "lat": lat, "lon": lon, "vmax_kt": actual["vmax_kt"]})
    return estado_desde_fila(actual, velocidad, rumbo), pronostico


def estado_desde_fila(fila, velocidad_kmh, rumbo):
    r34 = [fila.get(f"r34_{c}_km", np.nan) for c in ("ne", "se", "sw", "nw")]
    return {
        "lat": float(fila["lat"]),
        "lon": float(fila["lon"]),
        "vmax_kt": float(fila["vmax_kt"]),
        "velocidad_kmh": float(velocidad_kmh),
        "rumbo": float(rumbo),
        "r34_max_km": float(np.nanmax(r34)) if not np.all(np.isnan(r34)) else np.nan,
    }


# ------------------------------------------------------------------ NHC

def tormentas_activas_nhc():
    """Lista de tormentas activas según el NHC (Atlántico y Pacífico Oriental)."""
    r = requests.get(config.URL_NHC_TORMENTAS_ACTIVAS, timeout=30)
    r.raise_for_status()
    return r.json().get("activeStorms", [])


def descargar_zip(url):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return io.BytesIO(r.content)


def desde_nhc(zip_pronostico, r34_max_km=np.nan):
    """Lee el zip "5-day forecast track" del NHC y devuelve (estado, pronostico).

    zip_pronostico: ruta o BytesIO del archivo *_5day_*.zip.
    Los puntos traen TAU (horas), LAT, LON y MAXWIND (kt).
    """
    import geopandas as gpd

    if isinstance(zip_pronostico, (str, Path)):
        zip_pronostico = open(zip_pronostico, "rb")
    with zipfile.ZipFile(zip_pronostico) as z:
        nombre_pts = next(n for n in z.namelist() if n.lower().endswith("_pts.shp"))
    zip_pronostico.seek(0)
    puntos = gpd.read_file(zip_pronostico, layer=Path(nombre_pts).stem)
    columnas = {c.upper(): c for c in puntos.columns}
    pronostico = pd.DataFrame({
        "horas": pd.to_numeric(puntos[columnas["TAU"]]),
        "lat": pd.to_numeric(puntos[columnas["LAT"]]),
        "lon": pd.to_numeric(puntos[columnas["LON"]]),
        "vmax_kt": pd.to_numeric(puntos[columnas["MAXWIND"]]),
    }).sort_values("horas").reset_index(drop=True)
    nombre = str(puntos[columnas["STORMNAME"]].iloc[0]) if "STORMNAME" in columnas else "Tormenta"

    p0, p1 = pronostico.iloc[0], pronostico.iloc[1]
    dist = geo.distancia_km(p0["lat"], p0["lon"], p1["lat"], p1["lon"])
    rumbo = geo.rumbo_deg(p0["lat"], p0["lon"], p1["lat"], p1["lon"])
    velocidad = dist / max(p1["horas"] - p0["horas"], 1)
    estado = {
        "lat": float(p0["lat"]), "lon": float(p0["lon"]), "vmax_kt": float(p0["vmax_kt"]),
        "velocidad_kmh": float(velocidad), "rumbo": float(rumbo), "r34_max_km": float(r34_max_km),
    }
    return nombre, estado, pronostico
