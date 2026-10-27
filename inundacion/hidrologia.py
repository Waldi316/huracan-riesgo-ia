"""Lluvia -> escurrimiento -> caudal -> tirante, hora por hora, sobre el terreno.

Método (inspirado en el esquema HAND + curva de gasto sintética de la NOAA/OWP):

1. Escurrimiento: número de curva del SCS con la lluvia acumulada, por clase de cobertura.
2. Caudal en cada tramo de cauce: área drenada de cada clase x escurrimiento, retrasado
   con un hidrograma unitario triangular según el tiempo de concentración del tramo.
3. Curva de gasto sintética por tramo: para cada tirante h, las celdas con HAND < h forman
   la sección inundada; Manning da el caudal que esa sección conduce.
   El DEM no ve el lecho del río (marca la superficie del agua), así que a cada tramo se le
   suma la sección de cauce lleno de la geometría hidráulica regional: el agua se desborda
   solo cuando el caudal supera lo que el cauce conduce lleno.
4. Con el caudal de cada hora se interpola el tirante del tramo y se inunda toda celda
   cuya HAND quede por debajo: tirante en la celda = h - HAND.
5. Las lagunas costeras se simulan como vasos: reciben los ríos que les llegan y la lluvia
   directa, desaguan al mar por un vertedor y, al subir, inundan la planicie baja conectada.

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

def cauce_lleno(area_km2):
    """Ancho (m), tirante (m) y área (m2) del cauce lleno según el área drenada.

    Relaciones nacionales de EE. UU. de Bieger et al. (2015), las mismas que usa la NOAA/OWP
    para corregir las curvas de gasto HAND por la batimetría que el DEM no ve.
    """
    a = np.maximum(np.asarray(area_km2, dtype=float), 0.01)
    return 2.70 * a ** 0.352, 0.30 * a ** 0.213, 0.95 * a ** 0.53


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

    area_tramo = np.zeros(n_tramos)
    np.maximum.at(area_tramo, tramo_de_cauce, ter["area_drenada_km2"].ravel()[ter["cauce_plano"]])
    ancho_bf, tirante_bf, seccion_bf = cauce_lleno(area_tramo)

    orden = np.argsort(hand)
    hand_o, tramo_o = hand[orden], tramo_v[orden]
    caudal = np.zeros((n_tramos, len(niveles)))
    for k, h in enumerate(niveles):
        # Cauce principal: la sección llena más lo que sube el agua sobre su ancho.
        seccion_c = seccion_bf + ancho_bf * h
        caudal[:, k] = seccion_c * (seccion_c / (ancho_bf + 2 * tirante_bf)) ** (2 / 3) * np.sqrt(pend) / config.N_MANNING_CAUCE
        m = np.searchsorted(hand_o, h)
        if m == 0:
            continue
        volumen = np.bincount(tramo_o[:m], weights=(h - hand_o[:m]) * area_celda, minlength=n_tramos)
        superficie = np.bincount(tramo_o[:m], minlength=n_tramos) * area_celda
        seccion = volumen / longitud                     # m2
        ancho_sup = np.maximum(superficie / longitud, celda_m)
        radio = seccion / ancho_sup
        caudal[:, k] += seccion * radio ** (2 / 3) * np.sqrt(pend) / n
    caudal = np.maximum.accumulate(caudal, axis=1)       # monotónica en el tirante
    return niveles, caudal, tramo_celda


# ------------------------------------------------------------------ simulación

def simular(ter, lluvia_mm_h, lagunas=None):
    """Simula la inundación con una serie horaria de lluvia uniforme sobre el dominio.

    `lagunas`: {nombre: (lat, lon)} de un punto dentro de cada laguna costera (ver config.DOMINIOS).
    Devuelve un dict con: horas, caudal por tramo, tirante por tramo (m), nivel de cada laguna
    y lo necesario para reconstruir el tirante en cada celda con `tirante_en_celdas`.
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
    vasos = [_simular_laguna(ter, nombre, punto, tramo_de_cauce, caudal, lluvia.to_numpy())
             for nombre, punto in (lagunas or {}).items()]
    return {
        "horas": pd.DatetimeIndex(lluvia.index) if isinstance(lluvia.index, pd.DatetimeIndex) else np.arange(horas),
        "lluvia_mm_h": lluvia.to_numpy(),
        "caudal_m3s": caudal,
        "tirante_tramo_m": tirante,
        "tramo_celda": tramo_celda,
        "lagunas": vasos,
        "forma": ter["dem"].shape,
    }


# ------------------------------------------------------------------ lagunas costeras

def vaso_laguna(ter, punto, tramo_de_cauce):
    """Geometría del vaso: celdas de la laguna, planicie baja conectada y tramos que le llegan."""
    from scipy import ndimage

    from .exposicion import a_fila_columna

    if "agua_permanente" not in ter:
        raise ValueError("Las lagunas necesitan el agua de WorldCover en el terreno: vuelve a correr `terreno`.")
    fila, col = (int(v[0]) for v in a_fila_columna(ter, *punto))
    etiquetas, _ = ndimage.label(ter["agua_permanente"])
    if etiquetas[fila, col] == 0:
        raise ValueError(f"El punto {punto} no cae en agua de WorldCover.")
    vaso = etiquetas == etiquetas[fila, col]
    dem = ter["dem"]
    z0 = float(np.nanmedian(dem[vaso])) if np.isfinite(dem[vaso]).any() else 0.0

    # Planicie que se inundaría si la laguna sube hasta el máximo, conectada con ella.
    baja = np.isfinite(dem) & (dem < z0 + config.LAGUNA_SOBREELEVACION_MAX_M) & ~vaso
    componentes, _ = ndimage.label(baja | vaso, structure=np.ones((3, 3)))
    orilla = np.flatnonzero(((componentes == componentes[fila, col]) & baja).ravel())
    cota = np.clip(dem.ravel()[orilla] - z0, 0, None)

    # Ríos que desembocan: bocas (celdas de cauce sin otra de mayor área a su lado, donde termina el río)
    # a menos de ~300 m de la laguna; en cada grupo de bocas, el tramo con más área.
    area_cauce = np.where(ter["cauce"], ter["area_drenada_km2"], 0)
    boca = ter["cauce"] & (area_cauce >= ndimage.maximum_filter(area_cauce, size=3))
    junto = ndimage.binary_dilation(vaso, iterations=10) & ~vaso & boca
    grupos, n = ndimage.label(junto, structure=np.ones((3, 3)))
    posicion = np.full(dem.size, -1, dtype=np.int64)
    posicion[ter["cauce_plano"]] = np.arange(ter["cauce_plano"].size)
    area = area_cauce.ravel()
    tramos = set()
    for g in range(1, n + 1):
        celdas = np.flatnonzero((grupos == g).ravel())
        tramos.add(int(tramo_de_cauce[posicion[celdas[np.argmax(area[celdas])]]]))

    t = ter["transform"]
    area_celda = abs(float(t[0]) * float(t[4]))
    return {"orilla": orilla, "cota": cota, "tramos": sorted(tramos),
            "area_vaso_m2": vaso.sum() * area_celda, "area_celda_m2": area_celda}


def _simular_laguna(ter, nombre, punto, tramo_de_cauce, caudal, lluvia_mm_h, pasos_por_hora=6):
    geo = vaso_laguna(ter, punto, tramo_de_cauce)
    alturas = np.arange(0, config.LAGUNA_SOBREELEVACION_MAX_M, 0.01)
    cota = np.sort(geo["cota"])
    n = np.searchsorted(cota, alturas)
    acumulada = np.concatenate([[0.0], np.cumsum(cota)])
    volumenes = geo["area_vaso_m2"] * alturas + (n * alturas - acumulada[n]) * geo["area_celda_m2"]

    entrada = caudal[geo["tramos"]].sum(axis=0) + np.asarray(lluvia_mm_h) / 1000 / 3600 * geo["area_vaso_m2"]
    dt = 3600 / pasos_por_hora
    volumen, nivel = 0.0, np.zeros(len(entrada))
    for k, q_in in enumerate(entrada):
        for _ in range(pasos_por_hora):
            h = np.interp(volumen, volumenes, alturas)
            q_out = config.LAGUNA_COEF_VERTEDOR * config.LAGUNA_ANCHO_SALIDA_M * h ** 1.5
            volumen = max(volumen + (q_in - q_out) * dt, 0.0)
        nivel[k] = np.interp(volumen, volumenes, alturas)
    return {"nombre": nombre, "orilla": geo["orilla"], "cota": geo["cota"], "nivel_m": nivel,
            "entrada_m3s": entrada}


def _sin_agua_visible(ter):
    """Mar y cuerpos de agua permanentes (WorldCover 80): ahí no se dibuja inundación."""
    fuera = ter["mar"].ravel()
    if "cobertura" in ter:
        fuera = fuera | (ter["cobertura"].ravel() == 80)
    return fuera


def tirante_puntos(ter, sim, plano):
    """Tirante bruto (m, sin umbral) en las celdas `plano` para todas las horas: (celdas, horas)."""
    plano = np.asarray(plano)
    tramo = sim["tramo_celda"][plano]
    h = np.where(tramo[:, None] >= 0, sim["tirante_tramo_m"][np.clip(tramo, 0, None)], 0.0)
    d = h - ter["hand"].ravel()[plano][:, None]
    for laguna in sim.get("lagunas", []):
        orden = np.argsort(laguna["orilla"])
        i = np.clip(np.searchsorted(laguna["orilla"], plano, sorter=orden), 0, len(orden) - 1)
        dentro = laguna["orilla"][orden[i]] == plano
        d_lag = laguna["nivel_m"][None, :] - laguna["cota"][orden[i]][:, None]
        d = np.where(dentro[:, None], np.fmax(d, d_lag), d)
    return np.nan_to_num(np.clip(d, 0, None))


def _con_lagunas(sim, d, nivel):
    """Agrega a `d` (todas las celdas, en plano) el agua de cada laguna; `nivel(laguna)` da su nivel."""
    for laguna in sim.get("lagunas", []):
        d_lag = nivel(laguna) - laguna["cota"]
        d[laguna["orilla"]] = np.fmax(d[laguna["orilla"]], d_lag)
    return d


def tirante_en_celdas(ter, sim, k):
    """Tirante (m) en cada celda en la hora k. NaN donde no hay agua."""
    tramo = sim["tramo_celda"]
    h = np.where(tramo >= 0, sim["tirante_tramo_m"][np.clip(tramo, 0, None), k], 0.0)
    d = _con_lagunas(sim, h - ter["hand"].ravel(), lambda lag: lag["nivel_m"][k])
    d = np.where((d >= config.TIRANTE_MINIMO_VISIBLE_M) & ~_sin_agua_visible(ter), d, np.nan)
    return d.reshape(sim["forma"]).astype("float32")


def tirante_maximo(ter, sim):
    k = int(np.argmax(sim["tirante_tramo_m"].max(axis=0)))
    h = np.where(sim["tramo_celda"] >= 0, sim["tirante_tramo_m"].max(axis=1)[np.clip(sim["tramo_celda"], 0, None)], 0.0)
    d = _con_lagunas(sim, h - ter["hand"].ravel(), lambda lag: lag["nivel_m"].max())
    d = np.where((d >= config.TIRANTE_MINIMO_VISIBLE_M) & ~_sin_agua_visible(ter), d, np.nan)
    return d.reshape(sim["forma"]).astype("float32"), k
