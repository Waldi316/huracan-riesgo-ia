"""Lluvia -> escurrimiento -> caudal -> tirante, hora por hora, sobre el terreno.

Método (inspirado en el esquema HAND + curva de gasto sintética de la NOAA/OWP):

1. Escurrimiento: número de curva del SCS con la lluvia acumulada, por clase de cobertura.
2. Caudal en cada tramo de cauce: área drenada de cada clase x escurrimiento, retrasado
   con un hidrograma unitario triangular según el tiempo de concentración del tramo.
3. Curva de gasto sintética por tramo: para cada tirante h, las celdas con HAND < h forman
   la sección inundada; Manning da el caudal que esa sección conduce.
4. Con el caudal de cada hora se interpola el tirante del tramo y se inunda toda celda
   cuya HAND quede por debajo: tirante en la celda = h - HAND.

Es un modelo de primer orden: no simula el flujo 2D ni la marea de tormenta, y el
DEM de Copernicus es de superficie (incluye edificios y árboles).
"""
import numpy as np
import pandas as pd

from . import config

TAMANO_BLOQUE_TRAMO = 10      # celdas: los cauces se agrupan en tramos de ~300 m
VELOCIDAD_CAUCE_KMH = 5.4     # 1.5 m/s para el tiempo de concentración
TIEMPO_LADERA_H = 0.5


# ------------------------------------------------------------------ escurrimiento

def numero_curva(clases, humedad_iii=config.HUMEDAD_ANTECEDENTE_III):
    cn = np.array([config.NUMERO_CURVA.get(int(c), config.NUMERO_CURVA_DEFECTO) for c in clases], dtype=float)
    if humedad_iii:
        cn = 23 * cn / (10 + 0.13 * cn)
    return np.clip(cn, 30, 100)


def escurrimiento_scs(lluvia_mm_h, cn):
    """Escurrimiento horario (mm/h) para cada número de curva. Devuelve (n_cn, n_horas)."""
    p = np.cumsum(np.asarray(lluvia_mm_h, dtype=float))[None, :]
    s = 25400 / np.asarray(cn, dtype=float)[:, None] - 254
    ia = 0.2 * s
    q = np.where(p > ia, (p - ia) ** 2 / np.maximum(p - ia + s, 1e-9), 0.0)
    return np.diff(q, axis=1, prepend=0.0)


def hidrograma_unitario(tc_h, dt_h=1.0):
    """Hidrograma unitario triangular del SCS (suma 1)."""
    tp = 0.6 * tc_h + dt_h / 2
    tb = 2.67 * tp
    t = np.arange(dt_h / 2, tb + dt_h, dt_h)
    u = np.where(t <= tp, t / tp, np.clip((tb - t) / (tb - tp), 0, None))
    return u / u.sum()


def tiempo_concentracion_h(area_km2):
    """Ley de Hack para la longitud del cauce y velocidad constante en el cauce."""
    longitud_km = 1.609 * 1.4 * (np.asarray(area_km2) / 2.59) ** 0.6
    return TIEMPO_LADERA_H + longitud_km / VELOCIDAD_CAUCE_KMH


# ------------------------------------------------------------------ tramos y curvas de gasto

def tramos(ter):
    """Agrupa las celdas de cauce en tramos (bloque de ~300 m x clase de área drenada)."""
    alto, ancho = ter["dem"].shape
    plano = ter["cauce_plano"]
    fila, col = np.divmod(plano, ancho)
    bloque = (fila // TAMANO_BLOQUE_TRAMO) * (ancho // TAMANO_BLOQUE_TRAMO + 1) + col // TAMANO_BLOQUE_TRAMO
    area = ter["area_drenada_km2"].ravel()[plano]
    clase_area = np.floor(np.log2(np.maximum(area / config.UMBRAL_CAUCE_KM2, 1))).astype(int)
    _, tramo_de_cauce = np.unique(bloque * 64 + clase_area, return_inverse=True)
    return tramo_de_cauce


def curvas_de_gasto(ter, tramo_de_cauce, niveles=None, n=config.N_MANNING):
    """Caudal (m3/s) que conduce cada tramo para cada tirante. Devuelve (niveles, Q[n_tramos, n_niveles])."""
    if niveles is None:
        niveles = np.arange(*config.NIVELES_TIRANTE_M)
    t = ter["transform"]
    celda_m = abs(float(t[0]))
    area_celda = celda_m * abs(float(t[4]))
    n_tramos = tramo_de_cauce.max() + 1

    # Cada celda de terreno pertenece al tramo de la celda de cauce a la que drena.
    posicion = np.full(ter["dem"].size, -1, dtype=np.int64)
    posicion[ter["cauce_plano"]] = np.arange(ter["cauce_plano"].size)
    indice = ter["indice_cauce"].ravel()
    validas = indice >= 0
    tramo_celda = np.full(indice.size, -1, dtype=np.int64)
    tramo_celda[validas] = tramo_de_cauce[posicion[indice[validas]]]
    validas &= tramo_celda >= 0
    hand = ter["hand"].ravel()[validas].astype("float64")
    tramo_v = tramo_celda[validas]

    longitud = np.bincount(tramo_de_cauce, minlength=n_tramos) * celda_m
    pend = np.bincount(tramo_de_cauce, weights=ter["pendiente"].ravel()[ter["cauce_plano"]], minlength=n_tramos)
    pend = np.clip(pend / np.bincount(tramo_de_cauce, minlength=n_tramos), config.PENDIENTE_MINIMA, 0.1)

    orden = np.argsort(hand)
    hand_o, tramo_o = hand[orden], tramo_v[orden]
    caudal = np.zeros((n_tramos, len(niveles)))
    for k, h in enumerate(niveles):
        m = np.searchsorted(hand_o, h)
        if m == 0:
            continue
        volumen = np.bincount(tramo_o[:m], weights=(h - hand_o[:m]) * area_celda, minlength=n_tramos)
        superficie = np.bincount(tramo_o[:m], minlength=n_tramos) * area_celda
        seccion = volumen / longitud                     # m2
        ancho_sup = np.maximum(superficie / longitud, celda_m)
        radio = seccion / ancho_sup
        caudal[:, k] = seccion * radio ** (2 / 3) * np.sqrt(pend) / n
    caudal = np.maximum.accumulate(caudal, axis=1)       # monotónica en el tirante
    return niveles, caudal, tramo_celda


# ------------------------------------------------------------------ simulación

def simular(ter, lluvia_mm_h):
    """Simula la inundación con una serie horaria de lluvia uniforme sobre el dominio.

    Devuelve un dict con: horas, caudal por tramo, tirante por tramo (m) y lo necesario
    para reconstruir el tirante en cada celda con `tirante_en_celdas`.
    """
    lluvia = pd.Series(lluvia_mm_h).fillna(0.0)
    cn = numero_curva(ter["clases_cobertura"])
    q_mm_h = escurrimiento_scs(lluvia.to_numpy(), cn)                # (clases, horas)
    q_m_s = q_mm_h / 1000 / 3600

    tramo_de_cauce = tramos(ter)
    n_tramos = tramo_de_cauce.max() + 1
    area_clase = ter["area_clase_cauce_km2"].astype("float64") * 1e6  # (clases, celdas de cauce) en m2
    area_total = area_clase.sum(axis=0)
    # El caudal de un tramo es el de su celda con mayor área drenada (su salida).
    salida = pd.Series(area_total).groupby(tramo_de_cauce).idxmax().to_numpy()

    horas = len(lluvia)
    tc = np.round(tiempo_concentracion_h(area_total[salida] / 1e6) * 2) / 2   # agrupado cada 30 min
    caudal = np.zeros((n_tramos, horas))
    for valor in np.unique(tc):
        uh = hidrograma_unitario(valor)
        retrasado = np.stack([np.convolve(q, uh)[:horas] for q in q_m_s])  # (clases, horas)
        sel = np.flatnonzero(tc == valor)
        caudal[sel] = area_clase[:, salida[sel]].T @ retrasado

    niveles, gasto, tramo_celda = curvas_de_gasto(ter, tramo_de_cauce)
    tirante = np.zeros_like(caudal)
    for i in range(n_tramos):
        tirante[i] = np.interp(caudal[i], gasto[i], niveles)
    return {
        "horas": pd.DatetimeIndex(lluvia.index) if isinstance(lluvia.index, pd.DatetimeIndex) else np.arange(horas),
        "lluvia_mm_h": lluvia.to_numpy(),
        "caudal_m3s": caudal,
        "tirante_tramo_m": tirante,
        "tramo_celda": tramo_celda,
        "forma": ter["dem"].shape,
    }


def _sin_agua_visible(ter):
    """Mar y cuerpos de agua permanentes (WorldCover 80): ahí no se dibuja inundación."""
    fuera = ter["mar"].ravel()
    if "cobertura" in ter:
        fuera = fuera | (ter["cobertura"].ravel() == 80)
    return fuera


def tirante_en_celdas(ter, sim, k):
    """Tirante (m) en cada celda en la hora k. NaN donde no hay agua."""
    tramo = sim["tramo_celda"]
    h = np.where(tramo >= 0, sim["tirante_tramo_m"][np.clip(tramo, 0, None), k], 0.0)
    d = h - ter["hand"].ravel()
    d = np.where((d >= config.TIRANTE_MINIMO_VISIBLE_M) & ~_sin_agua_visible(ter), d, np.nan)
    return d.reshape(sim["forma"]).astype("float32")


def tirante_maximo(ter, sim):
    k = int(np.argmax(sim["tirante_tramo_m"].max(axis=0)))
    h = np.where(sim["tramo_celda"] >= 0, sim["tirante_tramo_m"].max(axis=1)[np.clip(sim["tramo_celda"], 0, None)], 0.0)
    d = h - ter["hand"].ravel()
    d = np.where((d >= config.TIRANTE_MINIMO_VISIBLE_M) & ~_sin_agua_visible(ter), d, np.nan)
    return d.reshape(sim["forma"]).astype("float32"), k
