"""Construcción del conjunto de entrenamiento a partir de huracanes históricos.

Para cada tormenta y cada hora de emisión (cada 6 h) se calcula lo que se
sabía en ese momento (posición, viento y pronóstico por persistencia) y se
etiqueta con lo que realmente pasó en las 72 h siguientes (radios de viento
de la mejor trayectoria).
"""
import numpy as np
import pandas as pd

from . import config, etiquetas, geo, ibtracs, indice_base, pronostico, variables


def emisiones_de_tormenta(trayectoria_horaria, paso_horas=config.PASO_EMISION_HORAS):
    tiempos = pd.to_datetime(trayectoria_horaria["tiempo"])
    return [t for t in tiempos if t.hour % paso_horas == 0 and t.minute == 0]


def filas_de_emision(tray, t0, ubicaciones, horizonte=config.HORIZONTE_HORAS):
    """Variables + índice base + etiqueta para todas las ubicaciones cercanas en una emisión."""
    resultado = pronostico.por_persistencia(tray, t0, horizonte=horizonte)
    if resultado is None:
        return None
    estado, pron = resultado
    cerca = geo.distancia_km(estado["lat"], estado["lon"], ubicaciones["lat"].to_numpy(), ubicaciones["lon"].to_numpy())
    cercanas = ubicaciones[cerca <= config.DISTANCIA_MAXIMA_EVALUACION_KM]
    if cercanas.empty:
        return None
    v = variables.calcular(estado, pron, cercanas, horizonte=horizonte)
    v = indice_base.calcular(v)
    viento = etiquetas.viento_maximo_recibido(
        tray, cercanas["lat"], cercanas["lon"], t_inicio=t0, t_fin=t0 + pd.Timedelta(hours=horizonte)
    )
    v["viento_recibido_kt"] = viento
    v["y34"] = (viento >= 34).astype(int)
    v["y64"] = (viento >= 64).astype(int)
    v["emision"] = t0
    return v


def construir(df_ibtracs, ubicaciones, sids=None, verbose=True):
    """Devuelve un DataFrame con una fila por (tormenta, emisión, ubicación)."""
    sids = sids if sids is not None else ibtracs.tormentas_relevantes(df_ibtracs)
    bloques = []
    for k, sid in enumerate(sids, 1):
        tormenta = df_ibtracs[df_ibtracs["sid"] == sid]
        if len(tormenta) < 3:
            continue
        tray = ibtracs.interpolar_horario(tormenta)
        for t0 in emisiones_de_tormenta(tray):
            filas = filas_de_emision(tray, t0, ubicaciones)
            if filas is not None:
                filas["sid"] = sid
                filas["temporada"] = int(tormenta["temporada"].iloc[0])
                filas["nombre_tormenta"] = tormenta["nombre"].iloc[0]
                bloques.append(filas)
        if verbose and k % 25 == 0:
            print(f"  {k}/{len(sids)} tormentas procesadas")
    if not bloques:
        raise ValueError("No se generó ninguna fila: revisa las tormentas y las ubicaciones")
    datos = pd.concat(bloques, ignore_index=True)
    columnas_float = [c for c in variables.COLUMNAS_MODELO if c in datos.columns]
    datos[columnas_float] = datos[columnas_float].astype(np.float32)
    return datos
