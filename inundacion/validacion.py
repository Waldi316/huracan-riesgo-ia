"""Validación de la inundación simulada contra agua observada por satélite (p. ej. Sentinel-1)."""
import numpy as np


def rasterizar(ter, ruta_vectorial):
    """Polígonos (shapefile, GeoPackage) -> máscara booleana en la malla del terreno."""
    import geopandas as gpd
    from rasterio.features import rasterize
    from rasterio.transform import Affine

    gdf = gpd.read_file(ruta_vectorial).to_crs(ter["crs"])
    forma = ter["dem"].shape
    return rasterize(((g, 1) for g in gdf.geometry if g is not None and not g.is_empty), out_shape=forma,
                     transform=Affine(*ter["transform"]), fill=0, dtype="uint8").astype(bool)


def metricas(simulada, observada, huella, tolerancia_celdas=1):
    """Aciertos, omisiones y falsas alarmas dentro de la huella observada por el satélite.

    Con tolerancia, una celda observada cuenta como acierto si hay agua simulada a esa
    distancia (y al revés para las falsas alarmas): absorbe el error de posición de 30 m.
    """
    from scipy import ndimage

    sim, obs = simulada & huella, observada & huella
    estructura = np.ones((3, 3), dtype=bool)
    sim_t = ndimage.binary_dilation(sim, estructura, iterations=tolerancia_celdas) if tolerancia_celdas else sim
    obs_t = ndimage.binary_dilation(obs, estructura, iterations=tolerancia_celdas) if tolerancia_celdas else obs
    aciertos = int((obs & sim_t).sum())
    omisiones = int((obs & ~sim_t).sum())
    falsas = int((sim & ~obs_t).sum())
    return {
        "aciertos": aciertos, "omisiones": omisiones, "falsas_alarmas": falsas,
        "pod": aciertos / max(aciertos + omisiones, 1),                 # agua observada que el modelo detecta
        "far": falsas / max(falsas + int((sim & obs_t).sum()), 1),      # agua simulada que no se observó
        "csi": aciertos / max(aciertos + omisiones + falsas, 1),
    }
