"""Utilidades geodésicas vectorizadas (sin arcpy).

Todas las distancias se calculan sobre la esfera (haversine), así que no
dependen de la proyección ni del factor "grados x 111.32", que sobrestima
las distancias este-oeste fuera del ecuador.
"""
import numpy as np

RADIO_TIERRA_KM = 6371.0088
KM_POR_MILLA_NAUTICA = 1.852


def distancia_km(lat1, lon1, lat2, lon2):
    """Distancia de gran círculo en km. Acepta escalares o arreglos (broadcast)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * RADIO_TIERRA_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def rumbo_deg(lat1, lon1, lat2, lon2):
    """Rumbo inicial de 1 hacia 2, en grados desde el norte (0-360)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def destino(lat, lon, rumbo, dist_km):
    """Punto al que se llega desde (lat, lon) avanzando dist_km con un rumbo dado."""
    lat, lon, rumbo = map(np.radians, (lat, lon, rumbo))
    d = np.asarray(dist_km) / RADIO_TIERRA_KM
    lat2 = np.arcsin(np.sin(lat) * np.cos(d) + np.cos(lat) * np.sin(d) * np.cos(rumbo))
    lon2 = lon + np.arctan2(np.sin(rumbo) * np.sin(d) * np.cos(lat), np.cos(d) - np.sin(lat) * np.sin(lat2))
    return np.degrees(lat2), (np.degrees(lon2) + 540) % 360 - 180


def cuadrante(rumbo):
    """Cuadrante (0=NE, 1=SE, 2=SW, 3=NW) del rumbo centro -> punto.

    Es el orden en que IBTrACS y el NHC reportan los radios de viento.
    """
    return (np.asarray(rumbo) // 90).astype(int) % 4


def diferencia_angular(a, b):
    """Diferencia firmada a - b en grados, en el rango (-180, 180]."""
    return (np.asarray(a) - np.asarray(b) + 180) % 360 - 180
