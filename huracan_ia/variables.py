"""Variables por ubicación a partir del estado actual y el pronóstico de la tormenta."""
import numpy as np
import pandas as pd

from . import config, geo

COLUMNAS_MODELO = [
    "dist_actual_km",
    "dist_min_km",
    "horas_a_minima",
    "vmax_actual_kt",
    "vmax_en_minima_kt",
    "lado_derecho",
    "margen_cono_km",
    "r34_max_km",
    "dist_min_menos_r34_km",
    "velocidad_kmh",
    "lat",
]


def _pronostico_horario(pronostico):
    horas = np.arange(pronostico["horas"].min(), pronostico["horas"].max() + 1, 1.0)
    return pd.DataFrame({
        "horas": horas,
        "lat": np.interp(horas, pronostico["horas"], pronostico["lat"]),
        "lon": np.interp(horas, pronostico["horas"], pronostico["lon"]),
        "vmax_kt": np.interp(horas, pronostico["horas"], pronostico["vmax_kt"]),
    })


def radio_cono_km(horas):
    claves = np.array(sorted(config.RADIO_CONO_MN))
    valores = np.array([config.RADIO_CONO_MN[k] for k in claves]) * geo.KM_POR_MILLA_NAUTICA
    return np.interp(horas, claves, valores)


def calcular(estado, pronostico, ubicaciones, horizonte=config.HORIZONTE_HORAS):
    """Devuelve un DataFrame con una fila por ubicación y las columnas de COLUMNAS_MODELO.

    ubicaciones: DataFrame con columnas id, lat, lon.
    """
    p = _pronostico_horario(pronostico[pronostico["horas"] <= horizonte])
    lats = ubicaciones["lat"].to_numpy(dtype=float)
    lons = ubicaciones["lon"].to_numpy(dtype=float)

    dist = geo.distancia_km(p["lat"].to_numpy()[:, None], p["lon"].to_numpy()[:, None], lats[None, :], lons[None, :])
    i_min = dist.argmin(axis=0)
    dist_min = dist[i_min, np.arange(len(lats))]
    horas_min = p["horas"].to_numpy()[i_min]

    # Dirección de avance en el punto más cercano (segmento i -> i+1)
    i_sig = np.minimum(i_min + 1, len(p) - 1)
    i_ant = np.where(i_sig == i_min, np.maximum(i_min - 1, 0), i_min)
    rumbo_avance = geo.rumbo_deg(p["lat"].to_numpy()[i_ant], p["lon"].to_numpy()[i_ant],
                                 p["lat"].to_numpy()[i_sig], p["lon"].to_numpy()[i_sig])
    rumbo_a_punto = geo.rumbo_deg(p["lat"].to_numpy()[i_min], p["lon"].to_numpy()[i_min], lats, lons)
    # +1 = a la derecha de la trayectoria (cuadrante más peligroso en el hemisferio norte)
    lado = np.sin(np.radians(geo.diferencia_angular(rumbo_a_punto, rumbo_avance)))
    lado = np.where(dist_min < 1.0, 0.0, lado)

    r34 = estado.get("r34_max_km", np.nan)
    return pd.DataFrame({
        "id": ubicaciones["id"].to_numpy(),
        "dist_actual_km": geo.distancia_km(estado["lat"], estado["lon"], lats, lons),
        "dist_min_km": dist_min,
        "horas_a_minima": horas_min,
        "vmax_actual_kt": estado["vmax_kt"],
        "vmax_en_minima_kt": p["vmax_kt"].to_numpy()[i_min],
        "lado_derecho": lado,
        "margen_cono_km": dist_min - radio_cono_km(horas_min),
        "r34_max_km": r34,
        "dist_min_menos_r34_km": dist_min - r34,
        "velocidad_kmh": estado["velocidad_kmh"],
        "lat": lats,
    })
