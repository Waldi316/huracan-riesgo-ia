"""Archivo histórico de pronósticos oficiales del NHC (Forecast/Advisory, TCM).

Cada aviso del NHC (cada 6 h: 03, 09, 15 y 21 UTC) trae la posición actual,
el viento máximo, los radios de 34 kt por cuadrante y el pronóstico oficial
a 12, 24, 36, 48, 60, 72, 96 y 120 h. Con ellos se entrena el modelo con la
misma información que habrá en operación, en lugar de la persistencia.

Fuente: https://www.nhc.noaa.gov/archive/ (texto de cada aviso).
"""
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from . import config, geo

URL_AVISO = "https://www.nhc.noaa.gov/archive/{anio}/{carpeta}/{atcf}.fstadv.{numero:03d}.shtml"

_MESES = {m: i for i, m in enumerate(
    ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"), 1)}
_COORD = r"(\d+\.\d)([NS])\s+(\d+\.\d)([EW])"
# "1500 UTC MON OCT 23 2023" (avisos recientes) o "0300Z SUN SEP 05 2004" (anteriores)
_RE_EMISION = re.compile(r"\n\s*(\d{2})(\d{2})(?: UTC|Z) \w{3} (\w{3}) (\d{1,2}) (\d{4})")
_RE_CENTRO = re.compile(r"CENTER LOCATED NEAR\s+" + _COORD + r"\s+AT (\d{2})/(\d{2})(\d{2})Z")
_RE_VIENTO = re.compile(r"MAX SUSTAINED WINDS\s+(\d+) KT")
_RE_MOVIMIENTO = re.compile(r"PRESENT MOVEMENT TOWARD.*?OR\s+(\d+) DEGREES AT\s+(\d+) KT", re.S)
_RE_R34 = re.compile(r"34 KT\.+\s*(\d+)NE\s+(\d+)SE\s+(\d+)SW\s+(\d+)NW")
_RE_PRONOSTICO = re.compile(r"(?:FORECAST|OUTLOOK) VALID (\d{2})/(\d{2})(\d{2})Z\s+" + _COORD + r"(.*?)(?=(?:FORECAST|OUTLOOK) VALID|\$\$|$)", re.S)
_RE_MAX = re.compile(r"MAX WIND\s+(\d+) KT")


def _carpeta(atcf):
    anio = int(atcf[-4:])
    # 2004 y 2005 guardan todos los avisos en una sola carpeta.
    return "mar" if anio <= 2005 else atcf[:4].lower()


def _latlon(lat, ns, lon, ew):
    return float(lat) * (1 if ns == "N" else -1), float(lon) * (-1 if ew == "W" else 1)


def _fecha(dia, hora, minuto, referencia):
    """Fecha completa a partir de DD/HHMMZ, usando el mes y año del aviso."""
    t = pd.Timestamp(year=referencia.year, month=referencia.month, day=1, hour=int(hora), minute=int(minuto))
    t = t + pd.Timedelta(days=int(dia) - 1)
    if int(dia) < referencia.day - 15:  # cambio de mes
        t = t + pd.DateOffset(months=1)
    return t


def leer_aviso(texto):
    """Extrae del texto de un aviso TCM el estado actual y el pronóstico. Devuelve dict o None."""
    m_emision = _RE_EMISION.search(texto)
    m_centro = _RE_CENTRO.search(texto)
    m_viento = _RE_VIENTO.search(texto)
    if not (m_emision and m_centro and m_viento):
        return None
    hh, mm, mes, dia, anio = m_emision.groups()
    emision = pd.Timestamp(year=int(anio), month=_MESES[mes], day=int(dia), hour=int(hh), minute=int(mm))
    lat, lon = _latlon(*m_centro.groups()[:4])
    t_centro = _fecha(*m_centro.groups()[4:], referencia=emision)

    # Los radios actuales están antes de la línea "REPEAT...CENTER".
    actual = texto[m_viento.start(): texto.find("REPEAT", m_viento.start())]
    r34 = _RE_R34.search(actual)
    mov = _RE_MOVIMIENTO.search(texto)
    filas = [{"horas": 0.0, "lat": lat, "lon": lon, "vmax_kt": float(m_viento.group(1))}]
    for m in _RE_PRONOSTICO.finditer(texto):
        t = _fecha(*m.groups()[:3], referencia=emision)
        plat, plon = _latlon(*m.groups()[3:7])
        vmax = _RE_MAX.search(m.group(8))
        if vmax:
            filas.append({"horas": (t - t_centro) / pd.Timedelta(hours=1), "lat": plat, "lon": plon,
                          "vmax_kt": float(vmax.group(1))})
    return {
        "tiempo": t_centro,
        "lat": lat,
        "lon": lon,
        "vmax_kt": float(m_viento.group(1)),
        "rumbo": float(mov.group(1)) if mov else np.nan,
        "velocidad_kmh": float(mov.group(2)) * geo.KM_POR_MILLA_NAUTICA if mov else 0.0,
        "r34_max_km": max(int(x) for x in r34.groups()) * geo.KM_POR_MILLA_NAUTICA if r34 else 0.0,
        "pronostico": pd.DataFrame(filas).drop_duplicates("horas").sort_values("horas").reset_index(drop=True),
    }


def _descargar_aviso(sesion, atcf, numero):
    url = URL_AVISO.format(anio=atcf[-4:], carpeta=_carpeta(atcf), atcf=atcf.lower(), numero=numero)
    r = sesion.get(url, timeout=60)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return re.sub(r"<[^>]+>", "", r.text)


def avisos_de_tormenta(atcf, sesion=None, maximo=150, faltantes_seguidos=3):
    """Lee todos los avisos de una tormenta (p. ej. EP182023), del 1 en adelante."""
    sesion = sesion or requests.Session()
    avisos, faltan = [], 0
    for numero in range(1, maximo + 1):
        texto = _descargar_aviso(sesion, atcf, numero)
        if texto is None:
            faltan += 1
            if faltan >= faltantes_seguidos:
                break
            continue
        faltan = 0
        aviso = leer_aviso(texto)
        if aviso:
            aviso["numero"] = numero
            avisos.append(aviso)
    return avisos


def _a_tabla(atcf, avisos):
    filas = []
    for a in avisos:
        for _, p in a["pronostico"].iterrows():
            filas.append({"atcf": atcf, "numero": a["numero"], "tiempo": a["tiempo"], "rumbo": a["rumbo"],
                          "velocidad_kmh": a["velocidad_kmh"], "r34_max_km": a["r34_max_km"], **p.to_dict()})
    return pd.DataFrame(filas)


def descargar(atcfs, carpeta=config.CARPETA_DATOS / "nhc_avisos", hilos=6):
    """Descarga (con caché por tormenta) los avisos de varias tormentas. Devuelve un DataFrame largo."""
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)

    def una(atcf):
        ruta = carpeta / f"{atcf}.csv"
        if not ruta.exists():
            tabla = _a_tabla(atcf, avisos_de_tormenta(atcf, requests.Session()))
            tabla.to_csv(ruta, index=False)
        return ruta

    with ThreadPoolExecutor(hilos) as ex:
        rutas = list(ex.map(una, atcfs))
    marcos = [pd.read_csv(r, parse_dates=["tiempo"]) for r in rutas if r.stat().st_size > 1]
    return pd.concat(marcos, ignore_index=True) if marcos else pd.DataFrame()


def emisiones(tabla, horizonte=config.HORIZONTE_HORAS):
    """Convierte la tabla de avisos en una lista de (tiempo, estado, pronostico) recortada al horizonte."""
    salida = []
    for (_, _), g in tabla.groupby(["atcf", "numero"]):
        g = g.sort_values("horas")
        a = g.iloc[0]
        if a["horas"] != 0 or len(g) < 2:
            continue
        pron = g[["horas", "lat", "lon", "vmax_kt"]].reset_index(drop=True)
        # Punto exacto en el horizonte (interpolado) para no perder las últimas horas.
        if pron["horas"].max() > horizonte:
            fila = {c: float(np.interp(horizonte, pron["horas"], pron[c])) for c in ("lat", "lon", "vmax_kt")}
            pron = pd.concat([pron[pron["horas"] < horizonte], pd.DataFrame([{"horas": float(horizonte), **fila}])],
                             ignore_index=True)
        estado = {"lat": float(a["lat"]), "lon": float(a["lon"]), "vmax_kt": float(a["vmax_kt"]),
                  "velocidad_kmh": float(a["velocidad_kmh"]), "rumbo": float(a["rumbo"]),
                  "r34_max_km": float(a["r34_max_km"])}
        salida.append((pd.Timestamp(a["tiempo"]), estado, pron))
    return salida
