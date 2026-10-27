"""Terreno: DEM de Copernicus (30 m), red de drenaje y altura sobre el cauce más cercano (HAND).

HAND (Height Above Nearest Drainage) dice cuántos metros tiene que subir el agua
del cauce al que drena una celda para alcanzarla. Es la base del método de
inundación de la NOAA (OWP) y funciona bien con DEM globales.
"""
import math
import os
from pathlib import Path

import numpy as np

from . import config

CRS_METRICO = "EPSG:32614"   # UTM 14N, cubre Guerrero y el centro del país
TAMANO_CELDA_M = 30.0


def _preparar_gdal():
    """GDAL lee los COG por HTTP; hereda el proxy y el bundle de certificados del entorno."""
    bundle = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
    if bundle and "CURL_CA_BUNDLE" not in os.environ:
        os.environ["CURL_CA_BUNDLE"] = bundle
    if os.environ.get("HTTPS_PROXY") and "GDAL_HTTP_PROXY" not in os.environ:
        os.environ["GDAL_HTTP_PROXY"] = os.environ["HTTPS_PROXY"]


def teselas_de_un_grado(dominio):
    """Esquinas suroeste (lat, lon enteros) de las teselas de 1° que cubren el dominio."""
    lats = range(math.floor(dominio["lat_min"]), math.floor(dominio["lat_max"]) + 1)
    lons = range(math.floor(dominio["lon_min"]), math.floor(dominio["lon_max"]) + 1)
    return [(la, lo) for la in lats for lo in lons]


def url_tesela(plantilla, lat, lon):
    return plantilla.format(ns="N" if lat >= 0 else "S", lat=abs(lat), ew="E" if lon >= 0 else "W", lon=abs(lon))


def leer_reproyectado(plantilla, dominio, ruta_salida, resampling="bilinear", tamano=TAMANO_CELDA_M, dtype="float32"):
    """Lee las teselas que cubren el dominio, las une y las reproyecta a UTM con celdas de `tamano` m."""
    import rasterio
    from rasterio.merge import merge
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    _preparar_gdal()
    fuentes = [rasterio.open("/vsicurl/" + url_tesela(plantilla, la, lo)) for la, lo in teselas_de_un_grado(dominio)]
    try:
        limites = (dominio["lon_min"], dominio["lat_min"], dominio["lon_max"], dominio["lat_max"])
        mosaico, transform = merge(fuentes, bounds=limites)
        crs_origen = fuentes[0].crs
    finally:
        for f in fuentes:
            f.close()
    alto, ancho = mosaico.shape[1:]
    destino_t, ancho_d, alto_d = calculate_default_transform(crs_origen, CRS_METRICO, ancho, alto, *limites,
                                                             resolution=tamano)
    destino = np.zeros((alto_d, ancho_d), dtype=dtype)
    reproject(mosaico[0], destino, src_transform=transform, src_crs=crs_origen, dst_transform=destino_t,
              dst_crs=CRS_METRICO, resampling=getattr(Resampling, resampling))
    perfil = {"driver": "GTiff", "height": alto_d, "width": ancho_d, "count": 1, "dtype": dtype,
              "crs": CRS_METRICO, "transform": destino_t, "compress": "deflate"}
    Path(ruta_salida).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(ruta_salida, "w", **perfil) as dst:
        dst.write(destino, 1)
    return ruta_salida


def descargar(nombre_dominio, carpeta=config.CARPETA_DATOS_INUNDACION):
    """DEM y cobertura de suelo del dominio, alineados en la misma malla UTM de 30 m."""
    dominio = config.DOMINIOS[nombre_dominio]
    carpeta = Path(carpeta)
    ruta_dem = carpeta / f"{nombre_dominio}_dem.tif"
    ruta_cob = carpeta / f"{nombre_dominio}_cobertura.tif"
    if not ruta_dem.exists():
        print("Descargando DEM Copernicus 30 m...")
        leer_reproyectado(config.URL_DEM_COPERNICUS, dominio, ruta_dem)
    if not ruta_cob.exists():
        print("Descargando ESA WorldCover...")
        # WorldCover viene en teselas de 3°; se alinea a la malla del DEM por vecino más cercano.
        leer_worldcover_alineado(dominio, ruta_dem, ruta_cob)
    return ruta_dem, ruta_cob


def leer_worldcover_alineado(dominio, ruta_dem, ruta_salida):
    import rasterio
    from rasterio.merge import merge
    from rasterio.warp import Resampling, reproject

    _preparar_gdal()
    esquinas = {(3 * math.floor(la / 3), 3 * math.floor(lo / 3)) for la, lo in teselas_de_un_grado(dominio)}
    fuentes = [rasterio.open("/vsicurl/" + url_tesela(config.URL_WORLDCOVER, la, lo)) for la, lo in sorted(esquinas)]
    try:
        limites = (dominio["lon_min"], dominio["lat_min"], dominio["lon_max"], dominio["lat_max"])
        mosaico, transform = merge(fuentes, bounds=limites)
        crs_origen = fuentes[0].crs
    finally:
        for f in fuentes:
            f.close()
    with rasterio.open(ruta_dem) as ref:
        perfil = ref.profile | {"dtype": "uint8"}
        destino = np.zeros((ref.height, ref.width), dtype="uint8")
        reproject(mosaico[0], destino, src_transform=transform, src_crs=crs_origen, dst_transform=ref.transform,
                  dst_crs=ref.crs, resampling=Resampling.mode)
    with rasterio.open(ruta_salida, "w", **perfil) as dst:
        dst.write(destino, 1)
    return ruta_salida


def procesar(ruta_dem, ruta_cobertura=None, umbral_cauce_km2=config.UMBRAL_CAUCE_KM2, ruta_salida=None):
    """Acondiciona el DEM y calcula dirección de flujo, área drenada, HAND y pendiente del cauce.

    Devuelve un dict de arreglos 2D más la georreferencia, y lo guarda en .npz.
    El mar (elevación <= 0 conectada al borde) queda fuera del cálculo.
    """
    import rasterio

    if not hasattr(np, "in1d"):   # pysheds 0.5 todavía usa np.in1d, que NumPy 2.4 eliminó
        np.in1d = np.isin
    from pysheds.grid import Grid
    from pysheds.sview import Raster

    with rasterio.open(ruta_dem) as src:
        crudo = src.read(1).astype("float64")
        transform, crs = src.transform, src.crs.to_string()
    mar = mascara_mar(crudo)

    # pysheds toma el nodata del archivo: se escribe una copia con el mar como nodata.
    ruta_tmp = Path(ruta_dem).with_suffix(".sin_mar.tif")
    with rasterio.open(ruta_dem) as src:
        perfil = src.profile | {"nodata": -9999.0, "dtype": "float64"}
    with rasterio.open(ruta_tmp, "w", **perfil) as dst:
        dst.write(np.where(mar, -9999.0, crudo), 1)
    grid = Grid.from_raster(str(ruta_tmp))
    dem = grid.read_raster(str(ruta_tmp))
    lleno = grid.resolve_flats(grid.fill_depressions(grid.fill_pits(dem)))
    fdir = grid.flowdir(lleno, nodata_out=np.int64(0))
    acc = grid.accumulation(fdir)
    area_celda_km2 = abs(transform.a * transform.e) / 1e6
    cauce = (np.asarray(acc) * area_celda_km2 >= umbral_cauce_km2) & ~mar
    # Con return_index pysheds devuelve, para cada celda, el índice plano de la celda de cauce a la que drena.
    indice = np.asarray(grid.compute_hand(fdir, lleno, Raster(cauce, viewfinder=acc.viewfinder), return_index=True))
    z = np.asarray(lleno, dtype="float64")
    hand = np.where(indice >= 0, z - z.ravel()[np.clip(indice, 0, None)], np.nan)
    pendiente = grid.cell_slopes(lleno, fdir)

    # Área drenada por clase de cobertura en cada celda de cauce: con ella se calcula el escurrimiento aguas arriba.
    cauce_plano = np.flatnonzero(cauce)
    clases, area_clase = np.array([0]), (np.asarray(acc) * area_celda_km2).ravel()[cauce_plano][None, :]
    if ruta_cobertura is not None:
        with rasterio.open(ruta_cobertura) as src:
            cobertura = src.read(1)
        agua_permanente = cobertura == 80        # antes de recortar el mar: incluye lagunas costeras
        cobertura = np.where(mar, 0, cobertura)
        clases = np.array(sorted(set(np.unique(cobertura[~mar]).tolist()) - {0}) or [0])
        area_clase = np.stack([
            np.asarray(grid.accumulation(fdir, weights=Raster((cobertura == c).astype("float64"), viewfinder=acc.viewfinder)))
            .ravel()[cauce_plano] * area_celda_km2
            for c in clases
        ])
    ruta_tmp.unlink(missing_ok=True)

    hand = np.where(mar, np.nan, np.clip(hand, 0, None))
    indice = np.where(mar | ~np.isfinite(hand), -1, indice).astype("int64")
    resultado = {
        "dem": np.where(mar, np.nan, crudo).astype("float32"),
        "hand": hand.astype("float32"),
        "indice_cauce": indice,
        "area_drenada_km2": np.where(mar, 0, np.asarray(acc) * area_celda_km2).astype("float32"),
        "pendiente": np.clip(np.nan_to_num(np.asarray(pendiente), nan=0.0), config.PENDIENTE_MINIMA, None).astype("float32"),
        "cauce": cauce,
        "mar": mar,
        "cauce_plano": cauce_plano,
        "clases_cobertura": clases,
        "area_clase_cauce_km2": area_clase.astype("float32"),
        "transform": np.array(transform)[:6],
        "crs": crs,
    }
    if ruta_cobertura is not None:
        resultado["cobertura"] = cobertura
        resultado["agua_permanente"] = agua_permanente
    if ruta_salida is not None:
        np.savez_compressed(ruta_salida, **resultado)
    return resultado


def mascara_mar(dem, nivel=0.5):
    """Celdas bajas (<= nivel) conectadas al borde del ráster: el mar en un DEM costero."""
    from scipy import ndimage

    bajo = ~(dem > nivel)
    etiquetas, _ = ndimage.label(bajo)
    borde = np.unique(np.concatenate([etiquetas[0], etiquetas[-1], etiquetas[:, 0], etiquetas[:, -1]]))
    return np.isin(etiquetas, borde[borde > 0])


def cargar(ruta_npz):
    datos = dict(np.load(ruta_npz, allow_pickle=False))
    datos["crs"] = str(datos["crs"])
    return datos
