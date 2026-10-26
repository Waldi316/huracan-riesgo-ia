"""Lluvia observada (CMORPH CDR horario, 0.25°) y elevación gruesa para el modelo de lluvia.

Cada archivo horario de CMORPH es global (~200 KB). Se descarga una vez, se
recorta a la caja de México y se guarda el recorte en caché como .npy.
"""
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from . import config

CARPETA_CMORPH = config.CARPETA_DATOS_INUNDACION / "cmorph"
RESOLUCION = 0.25
_CANDADO_HDF5 = threading.Lock()   # la librería HDF5 de netCDF4 no admite lecturas en paralelo


def malla_caja(caja=config.CAJA_LLUVIA):
    """Centros de celda de CMORPH dentro de la caja (lat ascendente, lon en -180..180)."""
    lats = np.arange(math.floor(caja["lat_min"] / RESOLUCION) * RESOLUCION + RESOLUCION / 2, caja["lat_max"], RESOLUCION)
    lons = np.arange(math.floor(caja["lon_min"] / RESOLUCION) * RESOLUCION + RESOLUCION / 2, caja["lon_max"], RESOLUCION)
    return lats, lons


def _ruta_cache(t, carpeta):
    return Path(carpeta) / f"{t:%Y}" / f"cmorph_{t:%Y%m%d%H}.npy"


def descargar_hora(t, caja=config.CAJA_LLUVIA, carpeta=CARPETA_CMORPH, sesion=None):
    """Lluvia (mm/h) de la hora t recortada a la caja. NaN donde CMORPH no tiene dato."""
    import netCDF4

    t = pd.Timestamp(t)
    ruta = _ruta_cache(t, carpeta)
    if ruta.exists():
        return np.load(ruta)
    for intento in range(3):
        try:
            r = (sesion or requests).get(config.URL_CMORPH_HORARIO.format(t=t), timeout=60)
            break
        except requests.ConnectionError:
            if intento == 2:
                raise
    if r.status_code == 404:
        return None
    r.raise_for_status()
    lats, lons = malla_caja(caja)
    with _CANDADO_HDF5, netCDF4.Dataset("cmorph.nc", memory=r.content) as ds:
        lat_arch, lon_arch = ds["lat"][:].data, ds["lon"][:].data
        i = np.abs(lat_arch[:, None] - lats[None, :]).argmin(axis=0)
        j = np.abs(lon_arch[:, None] - (lons % 360)[None, :]).argmin(axis=0)
        campo = ds["cmorph"][0, i.min():i.max() + 1, :]
        datos = np.ma.filled(campo[i - i.min()][:, j].astype("float32"), np.nan)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    np.save(ruta, datos)
    return datos


def descargar_horas(tiempos, hilos=16, **kwargs):
    """Descarga en paralelo. Devuelve {tiempo: arreglo} solo con las horas disponibles."""
    tiempos = sorted({pd.Timestamp(t) for t in tiempos})
    sesion = requests.Session()
    adaptador = requests.adapters.HTTPAdapter(pool_maxsize=hilos)
    sesion.mount("https://", adaptador)

    def una(t):
        try:
            return t, descargar_hora(t, sesion=sesion, **kwargs)
        except requests.RequestException as e:
            print(f"  sin CMORPH {t}: {e}")
            return t, None

    salida = {}
    with ThreadPoolExecutor(hilos) as ex:
        for i, (t, a) in enumerate(ex.map(una, tiempos), 1):
            if a is not None:
                salida[t] = a
            if i % 500 == 0:
                print(f"  CMORPH {i}/{len(tiempos)}")
    return salida


def serie_en_punto(lat, lon, inicio, fin, **kwargs):
    """Serie horaria (mm/h) de CMORPH en la celda que contiene (lat, lon)."""
    lats, lons = malla_caja()
    i, j = np.abs(lats - lat).argmin(), np.abs(lons - lon).argmin()
    horas = pd.date_range(pd.Timestamp(inicio).floor("h"), pd.Timestamp(fin).floor("h"), freq="1h")
    datos = descargar_horas(horas, **kwargs)
    return pd.Series([datos[t][i, j] if t in datos else np.nan for t in horas], index=horas, name="lluvia_mm_h")


# ------------------------------------------------------------------ elevación gruesa

def elevacion_caja(caja=config.CAJA_LLUVIA, carpeta=config.CARPETA_DATOS_INUNDACION):
    """Elevación media (m) en la malla de CMORPH, a partir de teselas de elevación de baja resolución.

    El mar queda en 0. Se usa para que el modelo de lluvia aprenda el efecto de la sierra.
    """
    ruta = Path(carpeta) / "elevacion_caja.npy"
    if ruta.exists():
        return np.load(ruta)
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject
    from rasterio.transform import from_origin

    from .terreno import _preparar_gdal

    _preparar_gdal()
    z = config.NIVEL_TESELA_ELEVACION
    xs = range(_tesela_x(caja["lon_min"], z), _tesela_x(caja["lon_max"], z) + 1)
    ys = range(_tesela_y(caja["lat_max"], z), _tesela_y(caja["lat_min"], z) + 1)
    lats, lons = malla_caja(caja)
    destino = np.zeros((len(lats), len(lons)), dtype="float32")
    # La malla de destino se arma con la latitud descendente (convención de los rásters) y luego se voltea.
    t_dest = from_origin(lons[0] - RESOLUCION / 2, lats[-1] + RESOLUCION / 2, RESOLUCION, RESOLUCION)
    for x in xs:
        for y in ys:
            with rasterio.open("/vsicurl/" + config.URL_ELEVACION_TESELA.format(z=z, x=x, y=y)) as src:
                parcial = np.full_like(destino, np.nan)
                reproject(src.read(1).astype("float32"), parcial, src_transform=src.transform, src_crs=src.crs,
                          dst_transform=t_dest, dst_crs="EPSG:4326", resampling=Resampling.average,
                          src_nodata=None, dst_nodata=np.nan)
                destino = np.where(np.isfinite(parcial), parcial, destino)
    destino = np.clip(destino[::-1], 0, None)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    np.save(ruta, destino)
    return destino


def _tesela_x(lon, z):
    return int((lon + 180) / 360 * 2 ** z)


def _tesela_y(lat, z):
    lat_r = math.radians(lat)
    return int((1 - math.asinh(math.tan(lat_r)) / math.pi) / 2 * 2 ** z)
