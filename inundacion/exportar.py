"""Salidas para SIG: NetCDF con dimensión de tiempo (Time Slider de ArcGIS Pro / QGIS),
GeoTIFF del tirante máximo y GeoPackage de inmuebles afectados."""
from pathlib import Path

import numpy as np
import pandas as pd

from . import hidrologia


def netcdf_tirante(ter, sim, ruta, horas=None):
    """Tirante horario (m) en un NetCDF CF: tiempo x y x x, en la proyección del DEM (UTM)."""
    import netCDF4
    from pyproj import CRS
    from rasterio.transform import Affine

    t = Affine(*ter["transform"])
    alto, ancho = sim["forma"]
    horas = range(len(sim["horas"])) if horas is None else horas
    with netCDF4.Dataset(ruta, "w") as nc:
        nc.Conventions = "CF-1.8"
        nc.title = "Tirante de inundación simulado por lluvia de ciclón tropical"
        nc.source = "huracan-riesgo-ia / inundacion (Copernicus DEM, ESA WorldCover, CMORPH, IBTrACS, NHC)"
        nc.createDimension("time", None)
        nc.createDimension("y", alto)
        nc.createDimension("x", ancho)
        vt = nc.createVariable("time", "f8", ("time",))
        vt.units, vt.calendar, vt.standard_name = "hours since 1970-01-01 00:00:00", "standard", "time"
        vx = nc.createVariable("x", "f8", ("x",))
        vy = nc.createVariable("y", "f8", ("y",))
        vx[:] = t.c + t.a * (np.arange(ancho) + 0.5)
        vy[:] = t.f + t.e * (np.arange(alto) + 0.5)
        vx.standard_name, vy.standard_name = "projection_x_coordinate", "projection_y_coordinate"
        vx.units = vy.units = "m"
        crs = nc.createVariable("crs", "i4")
        crs_obj = CRS.from_user_input(ter["crs"])
        for k, v in crs_obj.to_cf().items():
            setattr(crs, k, v)
        crs.esri_pe_string = crs_obj.to_wkt("WKT1_ESRI")
        vd = nc.createVariable("tirante", "f4", ("time", "y", "x"), zlib=True, complevel=4, fill_value=np.float32(np.nan),
                               chunksizes=(1, alto, ancho))
        vd.units, vd.long_name, vd.grid_mapping = "m", "tirante de agua sobre el terreno", "crs"
        epoca = pd.Timestamp("1970-01-01")
        for i, k in enumerate(horas):
            vt[i] = (pd.Timestamp(sim["horas"][k]) - epoca) / pd.Timedelta(hours=1)
            vd[i] = hidrologia.tirante_en_celdas(ter, sim, k)
    return ruta


def geotiff(ter, arreglo, ruta, nodata=np.nan):
    import rasterio
    from rasterio.transform import Affine

    perfil = {"driver": "GTiff", "height": arreglo.shape[0], "width": arreglo.shape[1], "count": 1,
              "dtype": "float32", "crs": ter["crs"], "transform": Affine(*ter["transform"]), "nodata": nodata,
              "compress": "deflate"}
    with rasterio.open(ruta, "w", **perfil) as dst:
        dst.write(arreglo.astype("float32"), 1)
    return ruta


def geopackage_inmuebles(res, ruta):
    import geopandas as gpd

    gdf = gpd.GeoDataFrame(res.copy(), geometry=gpd.points_from_xy(res["lon"], res["lat"]), crs="EPSG:4326")
    gdf["hora_inundacion"] = gdf["hora_inundacion"].astype(str).replace("NaT", "")
    gdf.to_file(ruta, layer="inmuebles", driver="GPKG")
    return ruta


def horas_con_agua(sim, margen=6):
    """Rango de horas desde que empieza la lluvia hasta que el agua baja (para no exportar horas vacías)."""
    activo = np.flatnonzero((sim["lluvia_mm_h"] > 0.1) | (sim["tirante_tramo_m"].max(axis=0) > 0.3))
    if activo.size == 0:
        return range(len(sim["horas"]))
    return range(max(activo[0] - margen, 0), min(activo[-1] + margen + 1, len(sim["horas"])))


def todo(ter, sim, res, carpeta, base):
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    maximo, _ = hidrologia.tirante_maximo(ter, sim)
    rutas = {
        "netcdf": netcdf_tirante(ter, sim, carpeta / f"{base}_tirante_horario.nc", horas_con_agua(sim)),
        "tirante_max": geotiff(ter, maximo, carpeta / f"{base}_tirante_max.tif"),
        "inmuebles": geopackage_inmuebles(res, carpeta / f"{base}_inmuebles.gpkg"),
    }
    res.to_csv(carpeta / f"{base}_inmuebles.csv", index=False)
    return rutas
