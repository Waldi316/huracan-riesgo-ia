"""Cartera de inmuebles dentro del dominio y su exposición a la inundación simulada."""
import numpy as np
import pandas as pd

from . import config, hidrologia

NIVELES = [(0.5, "Alto"), (config.TIRANTE_MINIMO_VISIBLE_M, "Medio")]


def a_lat_lon(ter, filas, columnas):
    from rasterio.transform import Affine, xy
    from rasterio.warp import transform

    t = Affine(*ter["transform"])
    x, y = xy(t, filas, columnas)
    lon, lat = transform(ter["crs"], "EPSG:4326", np.atleast_1d(x), np.atleast_1d(y))
    return np.asarray(lat), np.asarray(lon)


def a_fila_columna(ter, lat, lon):
    from rasterio.transform import Affine, rowcol
    from rasterio.warp import transform

    x, y = transform("EPSG:4326", ter["crs"], np.atleast_1d(lon), np.atleast_1d(lat))
    filas, cols = rowcol(Affine(*ter["transform"]), x, y)
    alto, ancho = ter["dem"].shape
    return np.clip(np.asarray(filas), 0, alto - 1), np.clip(np.asarray(cols), 0, ancho - 1)


def cartera_simulada(ter, n=250, semilla=7, elevacion_maxima=150):
    """Inmuebles FICTICIOS: celdas de tierra elegidas al azar, con más peso en zona urbana (WorldCover 50)."""
    rng = np.random.default_rng(semilla)
    dem = ter["dem"]
    cob = ter.get("cobertura", np.zeros(dem.shape, dtype="uint8"))
    candidatas = np.flatnonzero(np.isfinite(dem).ravel() & (dem.ravel() > 1) & (dem.ravel() < elevacion_maxima)
                                & np.isfinite(ter["hand"].ravel()) & ~ter["cauce"].ravel())
    candidatas = candidatas[cob.ravel()[candidatas] != 80]   # fuera de cuerpos de agua
    peso = np.where(cob.ravel()[candidatas] == 50, 20.0, 1.0)
    elegidas = rng.choice(candidatas, size=n, replace=False, p=peso / peso.sum())
    filas, cols = np.divmod(elegidas, dem.shape[1])
    lat, lon = a_lat_lon(ter, filas, cols)
    return pd.DataFrame({"id": [f"SIM-INU-{i + 1:04d}" for i in range(n)],
                         "nombre": [f"Inmueble simulado {i + 1}" for i in range(n)],
                         "lat": lat, "lon": lon})


def exposicion(ter, sim, cartera):
    """Tirante por hora en cada inmueble, máximo, hora en que se inunda y nivel."""
    filas, cols = a_fila_columna(ter, cartera["lat"].to_numpy(), cartera["lon"].to_numpy())
    plano = filas * ter["dem"].shape[1] + cols
    tramo = sim["tramo_celda"][plano]
    hand = ter["hand"].ravel()[plano]
    h = np.where(tramo[:, None] >= 0, sim["tirante_tramo_m"][np.clip(tramo, 0, None)], 0.0)
    tirante = np.nan_to_num(np.clip(h - hand[:, None], 0, None))        # (inmuebles, horas); sin HAND = seco
    tirante[tirante < config.TIRANTE_MINIMO_VISIBLE_M] = 0.0
    res = cartera.copy()
    res["elevacion_m"] = ter["dem"].ravel()[plano]
    res["hand_m"] = hand
    res["tirante_max_m"] = tirante.max(axis=1)
    inundado = tirante > 0
    horas = sim["horas"]
    primera = np.where(inundado.any(axis=1), inundado.argmax(axis=1), -1)
    res["hora_inundacion"] = [horas[k] if k >= 0 else pd.NaT for k in primera]
    res["nivel"] = nivel(res["tirante_max_m"])
    return res, tirante


def nivel(tirante_m):
    t = np.asarray(tirante_m)
    return np.select([t >= u for u, _ in NIVELES], [n for _, n in NIVELES], default="Sin afectación")


def resumen(res):
    return res["nivel"].value_counts().reindex(["Alto", "Medio", "Sin afectación"], fill_value=0).to_dict()


def area_inundada_km2(ter, sim):
    """Área con tirante visible en cada hora (km2)."""
    t = ter["transform"]
    area_celda = abs(float(t[0]) * float(t[4])) / 1e6
    return np.array([np.isfinite(hidrologia.tirante_en_celdas(ter, sim, k)).sum() * area_celda
                     for k in range(len(sim["horas"]))])
