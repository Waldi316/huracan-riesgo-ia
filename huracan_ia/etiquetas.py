"""Etiquetas de entrenamiento: qué viento recibió realmente cada ubicación.

Se usan los radios de viento por cuadrante de la mejor trayectoria (IBTrACS):
una ubicación "recibió" vientos de 34 kt en una hora si su distancia al centro
es menor que el radio de 34 kt del cuadrante en el que se encuentra.
"""
import numpy as np

from . import geo

CUADRANTES = ("ne", "se", "sw", "nw")


def viento_maximo_recibido(trayectoria_horaria, lats, lons, t_inicio=None, t_fin=None):
    """Umbral máximo (0, 34, 50 o 64 kt) que alcanzó cada ubicación en la ventana.

    trayectoria_horaria: salida de ibtracs.interpolar_horario.
    t_inicio es exclusivo y t_fin inclusivo, para no contar la hora de emisión.
    """
    t = trayectoria_horaria
    if t_inicio is not None:
        t = t[t["tiempo"] > t_inicio]
    if t_fin is not None:
        t = t[t["tiempo"] <= t_fin]
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    resultado = np.zeros(len(lats), dtype=int)
    if t.empty or len(lats) == 0:
        return resultado

    # Matrices horas x ubicaciones
    clat = t["lat"].to_numpy()[:, None]
    clon = t["lon"].to_numpy()[:, None]
    dist = geo.distancia_km(clat, clon, lats[None, :], lons[None, :])
    cuad = geo.cuadrante(geo.rumbo_deg(clat, clon, lats[None, :], lons[None, :]))

    for umbral in (34, 50, 64):
        radios = np.stack([t[f"r{umbral}_{c}_km"].to_numpy() for c in CUADRANTES], axis=1)  # horas x 4
        radio_en_punto = np.take_along_axis(radios, cuad, axis=1)
        alcanzado = (dist <= radio_en_punto) & (radio_en_punto > 0)
        resultado[alcanzado.any(axis=0)] = umbral
    return resultado
