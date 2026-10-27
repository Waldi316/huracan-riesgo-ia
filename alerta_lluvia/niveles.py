"""Acumulado de 3 días, percentil propio de cada punto y nivel de alerta.

Umbrales calibrados contra órdenes de mantenimiento por agua (jun-oct 2026): la tasa de
orden se duplica cuando la lluvia de 3 días pasa de 50 mm o del percentil 90 del propio
punto. Son provisionales (AUC 0.58) hasta tener órdenes con fecha y causa reales.
"""
import numpy as np
import pandas as pd

DIAS_ACUMULADO = 3
UMBRAL_ALERTA_MM = 50.0
UMBRAL_VIGILANCIA_MM = 25.0
PERCENTIL_PROPIO = 90
MINIMO_PERCENTIL_MM = 20.0   # en zonas secas el P90 puede ser 2 mm; no se alerta por eso
DIAS_MINIMOS_PERCENTIL = 60

ALERTA, VIGILANCIA, SIN_ALERTA = "Alerta", "Vigilancia", "Sin alerta"


def acumulado(diaria, dias=DIAS_ACUMULADO):
    """Suma móvil de `dias` días. `diaria`: DataFrame (fechas x puntos) en mm/día.

    Un día faltante deja el acumulado en NaN en lugar de subestimarlo.
    """
    diaria = diaria.sort_index().asfreq("D")
    return diaria.rolling(dias, min_periods=dias).sum()


def percentil_propio(acum, percentil=PERCENTIL_PROPIO, dias_minimos=DIAS_MINIMOS_PERCENTIL):
    """Percentil del acumulado de cada punto con su histórico. NaN si hay pocos días."""
    p = acum.quantile(percentil / 100)
    return p.where(acum.notna().sum() >= dias_minimos)


def clasificar(lluvia_acum, p_propio):
    """Nivel por punto. Alerta: >= 50 mm, o >= su P90 con al menos 20 mm. Vigilancia: >= 25 mm."""
    lluvia_acum = np.asarray(lluvia_acum, dtype=float)
    p_propio = np.asarray(p_propio, dtype=float)
    por_percentil = (lluvia_acum >= np.nan_to_num(p_propio, nan=np.inf)) & (lluvia_acum >= MINIMO_PERCENTIL_MM)
    nivel = np.full(lluvia_acum.shape, SIN_ALERTA, dtype=object)
    nivel[lluvia_acum >= UMBRAL_VIGILANCIA_MM] = VIGILANCIA
    nivel[(lluvia_acum >= UMBRAL_ALERTA_MM) | por_percentil] = ALERTA
    return nivel


def tabla_del_dia(diaria, fecha=None):
    """Tabla por punto para `fecha` (por omisión, el último día con dato completo)."""
    acum = acumulado(diaria)
    if fecha is None:
        fecha = acum.dropna(how="all").index.max()
    fecha = pd.Timestamp(fecha)
    p90 = percentil_propio(acum.loc[:fecha])
    tabla = pd.DataFrame({
        "lluvia_24h": diaria.sort_index().asfreq("D").loc[fecha],
        f"lluvia_{24 * DIAS_ACUMULADO}h": acum.loc[fecha],
        f"p{PERCENTIL_PROPIO}_{24 * DIAS_ACUMULADO}h": p90,
    })
    tabla["nivel"] = clasificar(tabla.iloc[:, 1], tabla.iloc[:, 2])
    tabla["fecha_dato"] = fecha.date()
    return tabla
