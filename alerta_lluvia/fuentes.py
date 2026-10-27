"""Lluvia diaria CMORPH ICDR (NOAA/CPC): 0.25°, mm/día, latencia de 1 día, sin cuenta.

Cada archivo es global. Se baja, se toma el valor de la celda de cada punto y se guarda
en caché por día para no volver a bajarlo.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import requests

URL_ICDR_DIARIO = ("https://ftp.cpc.ncep.noaa.gov/precip/CMORPH_RT/ICDR/0.25deg-DLY_00Z/"
                   "CMORPH_V0.x_ADJ_0.25deg-DLY_00Z_{f:%Y%m%d}.nc")   # desde 2026-03-01
CARPETA_CACHE = Path("datos/alerta_lluvia/icdr")


def lluvia_dia(fecha, lats, lons, carpeta=CARPETA_CACHE, url=URL_ICDR_DIARIO):
    """mm/día en la celda de cada punto (lon en -180..180). None si el día aún no se publica."""
    import netCDF4

    fecha = pd.Timestamp(fecha)
    ruta = Path(carpeta) / f"{fecha:%Y%m%d}.npz"
    if ruta.exists():
        guardado = np.load(ruta)
        return _en_puntos(guardado["campo"], guardado["lat"], guardado["lon"], lats, lons)
    r = requests.get(url.format(f=fecha), timeout=120)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    with netCDF4.Dataset("icdr.nc", memory=r.content) as ds:
        lat, lon = ds["lat"][:].data, ds["lon"][:].data
        campo = np.ma.filled(ds["cmorph"][0].astype("float32"), np.nan)   # int16 empaquetado; netCDF4 aplica la escala
    ruta.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ruta, campo=campo, lat=lat, lon=lon)
    return _en_puntos(campo, lat, lon, lats, lons)


def _en_puntos(campo, lat, lon, lats, lons):
    i = np.abs(lat[:, None] - np.asarray(lats)[None, :]).argmin(axis=0)
    j = np.abs(lon[:, None] - (np.asarray(lons) % 360)[None, :]).argmin(axis=0)
    return campo[i, j]


def serie_diaria(puntos, inicio, fin, **kwargs):
    """DataFrame (fechas x puntos) en mm/día. `puntos`: DataFrame con columnas lat, lon."""
    filas = {}
    for f in pd.date_range(inicio, fin, freq="D"):
        v = lluvia_dia(f, puntos["lat"], puntos["lon"], **kwargs)
        if v is not None:
            filas[f] = v
    return pd.DataFrame.from_dict(filas, orient="index", columns=puntos.index)
