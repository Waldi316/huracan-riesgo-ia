"""Construcción del conjunto de entrenamiento a partir de huracanes históricos.

Para cada tormenta y cada hora de emisión (cada 6 h) se calcula lo que se
sabía en ese momento (posición, viento y pronóstico por persistencia) y se
etiqueta con lo que realmente pasó en las 72 h siguientes (radios de viento
de la mejor trayectoria).
"""
import numpy as np
import pandas as pd

from . import config, etiquetas, geo, ibtracs, indice_base, nhc_archivo, pronostico, variables


def emisiones_de_tormenta(trayectoria_horaria, paso_horas=config.PASO_EMISION_HORAS):
    tiempos = pd.to_datetime(trayectoria_horaria["tiempo"])
    return [t for t in tiempos if t.hour % paso_horas == 0 and t.minute == 0]


def filas_de_emision(tray, t0, ubicaciones, horizonte=config.HORIZONTE_HORAS, estado_pron=None):
    """Variables + índice base + etiqueta para todas las ubicaciones cercanas en una emisión.

    estado_pron: (estado, pronostico) oficial del NHC; si no se da, se usa la persistencia.
    """
    resultado = estado_pron or pronostico.por_persistencia(tray, t0, horizonte=horizonte)
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


def construir_con_nhc(df_ibtracs, ubicaciones, tabla_avisos, sids=None, verbose=True):
    """Como construir, pero cada emisión es un aviso real del NHC con su pronóstico oficial.

    Devuelve (oficial, persistencia): dos conjuntos con las mismas emisiones y
    ubicaciones, uno con las variables del pronóstico oficial y otro con las de
    la persistencia, para comparar ambos en igualdad de condiciones.
    """
    sids = sids if sids is not None else ibtracs.tormentas_relevantes(df_ibtracs)
    atcf_por_sid = ibtracs.atcf_de_tormentas(df_ibtracs, sids)
    oficial, persistencia = [], []
    for k, sid in enumerate(sids, 1):
        tormenta = df_ibtracs[df_ibtracs["sid"] == sid]
        avisos = tabla_avisos[tabla_avisos["atcf"].isin(atcf_por_sid.get(sid, []))]
        if len(tormenta) < 3 or avisos.empty:
            continue
        tray = ibtracs.interpolar_horario(tormenta)
        tiempos = set(tray["tiempo"])
        for t0, estado, pron in nhc_archivo.emisiones(avisos):
            if t0 not in tiempos:
                continue
            filas_p = filas_de_emision(tray, t0, ubicaciones)
            if filas_p is None:
                continue
            filas_o = filas_de_emision(tray, t0, ubicaciones, estado_pron=(estado, pron))
            if filas_o is None:
                continue
            # Mismas ubicaciones en ambos (la cercanía se mide desde posiciones casi iguales).
            comunes = set(filas_o["id"]) & set(filas_p["id"])
            for filas, destino in ((filas_o, oficial), (filas_p, persistencia)):
                filas = filas[filas["id"].isin(comunes)].copy()
                filas["sid"] = sid
                filas["temporada"] = int(tormenta["temporada"].iloc[0])
                filas["nombre_tormenta"] = tormenta["nombre"].iloc[0]
                destino.append(filas)
        if verbose and k % 25 == 0:
            print(f"  {k}/{len(sids)} tormentas procesadas")
    if not oficial:
        raise ValueError("No se generó ninguna fila con avisos del NHC")
    salida = []
    for bloques in (oficial, persistencia):
        datos = pd.concat(bloques, ignore_index=True)
        columnas_float = [c for c in variables.COLUMNAS_MODELO if c in datos.columns]
        datos[columnas_float] = datos[columnas_float].astype(np.float32)
        salida.append(datos.sort_values(["sid", "emision", "id"]).reset_index(drop=True))
    return tuple(salida)
