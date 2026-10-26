"""Índice de riesgo con pesos fijos: la línea base contra la que se compara el modelo.

Reproduce la lógica de un proceso operativo típico de alertamiento:
cuatro variables de amenaza escalonadas (ubicación en el cono, viento,
distancia al pronóstico y distancia a la posición observada), ponderadas,
y combinadas con la vulnerabilidad del inmueble.
"""
import numpy as np
import pandas as pd

from . import config


def _escalonar_distancia(dist, tabla, valor_lejano=0.2):
    resultado = np.full(len(dist), valor_lejano)
    for limite, valor in reversed(tabla):
        resultado = np.where(dist < limite, valor, resultado)
    return resultado


def _escalonar_viento(vmax):
    resultado = np.full(len(vmax), 0.2)
    for minimo, maximo, valor in config.TABLA_VIENTO_KT:
        resultado = np.where((vmax >= minimo) & (vmax <= maximo), valor, resultado)
    return resultado


def calcular(variables, vulnerabilidad=None):
    """Agrega ifa, riesgo y nivel_base al DataFrame de variables.

    Solo se consideran "afectadas" las ubicaciones dentro del cono o del
    buffer de la trayectoria; las demás quedan con ifa = 0.
    """
    v = variables
    dentro_cono = v["margen_cono_km"].to_numpy() <= 0
    afectada = dentro_cono | (v["dist_min_km"].to_numpy() <= config.BUFFER_TRAYECTORIA_KM)

    v1 = np.where(dentro_cono, 1.0, 0.5)
    v2 = _escalonar_viento(v["vmax_en_minima_kt"].to_numpy())
    v3 = _escalonar_distancia(v["dist_min_km"].to_numpy(), config.TABLA_DIST_PRONOSTICO_KM)
    v4 = _escalonar_distancia(v["dist_actual_km"].to_numpy(), config.TABLA_DIST_OBSERVADO_KM)
    w = config.PESOS_AMENAZA
    ifa = v1 * w["ubicacion_cono"] + v2 * w["viento"] + v3 * w["dist_pronostico"] + v4 * w["dist_observado"]
    ifa = np.where(afectada, ifa, 0.0)

    if vulnerabilidad is None:
        vulnerabilidad = np.full(len(v), 0.5)
    riesgo = np.where(afectada, ifa * config.PESO_AMENAZA + np.asarray(vulnerabilidad) * config.PESO_VULNERABILIDAD, 0.0)
    nivel = pd.cut(riesgo, [-np.inf, config.CORTE_BAJO, config.CORTE_ALTO - 1e-9, np.inf], labels=["Bajo", "Medio", "Alto"])

    salida = v.copy()
    salida["ifa"] = ifa
    salida["riesgo_base"] = riesgo
    salida["nivel_base"] = np.where(afectada, nivel.astype(str), "Sin afectación")
    return salida
