"""Modelo de lluvia de ciclones tropicales: LightGBM contra el modelo paramétrico R-CLIPER.

Pregunta que responde: dada la posición, intensidad y movimiento de la tormenta,
¿cuántos mm/h van a caer en un punto? Se entrena con la lluvia observada por
satélite (CMORPH) en todas las tormentas que pasaron cerca de México desde 2004,
y se evalúa con temporadas posteriores, que el modelo nunca vio.

R-CLIPER (Tuleya et al. 2007, Lonfat et al. 2007) es el modelo climatológico que
usa el NHC como referencia: un perfil radial simétrico que solo depende del
viento máximo. No sabe nada de la sierra ni del lado de la tormenta.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from huracan_ia import geo, ibtracs

from . import config, lluvia

VARIABLES = [
    "dist_km", "vmax_kt", "dv12_kt", "velocidad_kmh", "sin_ang", "cos_ang", "lat_centro", "lon_centro",
    "elev_celda", "tierra_celda", "elev_centro", "tierra_centro", "ascenso_orografico",
    # Lo que se observaba al emitir el pronóstico: qué tan "lluviosa" venía la tormenta y el plazo.
    "plazo_h", "lluvia_nucleo_emision", "lluvia_anillo_emision", "lluvia_celda_emision",
]
INFLOW_DEG = 20.0   # ángulo con el que el viento entra hacia el centro
PLAZOS_H = np.arange(6, 73, 6)   # el pronóstico se emite al menos 6 h antes de la hora que predice
RADIO_NUCLEO_KM = 150
RADIO_ANILLO_KM = 400


# ------------------------------------------------------------------ estado de la tormenta

def estado_horario(trayectoria):
    """De una trayectoria (tiempo, lat, lon, vmax_kt) a una tabla horaria con movimiento e intensificación."""
    t = trayectoria[["tiempo", "lat", "lon", "vmax_kt"]].copy()
    for col, valor in (("sid", "X"), ("temporada", 0), ("nombre", "X")):
        t[col] = trayectoria[col].iloc[0] if col in trayectoria else valor
    h = ibtracs.interpolar_horario(t)
    previo = h.shift(6)
    h["velocidad_kmh"] = geo.distancia_km(previo["lat"], previo["lon"], h["lat"], h["lon"]) / 6
    h["rumbo"] = geo.rumbo_deg(previo["lat"], previo["lon"], h["lat"], h["lon"])
    h["dv12_kt"] = h["vmax_kt"] - h["vmax_kt"].shift(12)
    # Las primeras horas no tienen historia: se rellenan con el primer valor calculable.
    h[["velocidad_kmh", "rumbo"]] = h[["velocidad_kmh", "rumbo"]].bfill()
    h["dv12_kt"] = h["dv12_kt"].fillna(0.0)
    return h[["tiempo", "lat", "lon", "vmax_kt", "velocidad_kmh", "rumbo", "dv12_kt"]]


class Orografia:
    """Elevación gruesa en la malla de CMORPH y su gradiente, con muestreo por vecino más cercano."""

    def __init__(self, elevacion=None, caja=config.CAJA_LLUVIA):
        self.lats, self.lons = lluvia.malla_caja(caja)
        self.z = lluvia.elevacion_caja(caja) if elevacion is None else elevacion
        km_lat = lluvia.RESOLUCION * 111.2
        km_lon = km_lat * np.cos(np.radians(self.lats))[:, None]
        dz_dlat, dz_dlon = np.gradient(self.z)
        self.dz_norte = dz_dlat / km_lat     # m por km hacia el norte
        self.dz_este = dz_dlon / km_lon

    def indices(self, lat, lon):
        i = np.clip(np.round((np.asarray(lat) - self.lats[0]) / lluvia.RESOLUCION).astype(int), 0, len(self.lats) - 1)
        j = np.clip(np.round((np.asarray(lon) - self.lons[0]) / lluvia.RESOLUCION).astype(int), 0, len(self.lons) - 1)
        return i, j


def contexto_emision(campo, estado_emision, oro, lat, lon):
    """Lluvia observada (CMORPH) en la hora de emisión: núcleo, anillo exterior y en el propio punto."""
    lat_m, lon_m = np.meshgrid(oro.lats, oro.lons, indexing="ij")
    dist = geo.distancia_km(estado_emision["lat"], estado_emision["lon"], lat_m, lon_m)
    nucleo = campo[dist <= RADIO_NUCLEO_KM]
    anillo = campo[(dist > RADIO_NUCLEO_KM) & (dist <= RADIO_ANILLO_KM)]
    i, j = oro.indices(lat, lon)
    return {
        "lluvia_nucleo_emision": float(np.nanmean(nucleo)) if np.isfinite(nucleo).any() else np.nan,
        "lluvia_anillo_emision": float(np.nanmean(anillo)) if np.isfinite(anillo).any() else np.nan,
        "lluvia_celda_emision": campo[i, j],
    }


def variables(estado, lat, lon, oro, contexto=None):
    """Variables para cada par (estado de tormenta, punto). `estado` es un dict o una fila.

    contexto: dict con plazo_h y la lluvia observada en la emisión (ver `contexto_emision`).
    """
    contexto = contexto or {}
    lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    dist = geo.distancia_km(estado["lat"], estado["lon"], lat, lon)
    rumbo_punto = geo.rumbo_deg(estado["lat"], estado["lon"], lat, lon)
    ang = np.radians(geo.diferencia_angular(rumbo_punto, estado["rumbo"]))
    i, j = oro.indices(lat, lon)
    ic, jc = oro.indices(estado["lat"], estado["lon"])
    # Circulación ciclónica (antihoraria) con entrada hacia el centro: hacia dónde sopla el viento en el punto.
    hacia = np.radians(rumbo_punto - 90 - INFLOW_DEG)
    ascenso = oro.dz_norte[i, j] * np.cos(hacia) + oro.dz_este[i, j] * np.sin(hacia)
    return pd.DataFrame({
        "dist_km": dist,
        "vmax_kt": estado["vmax_kt"],
        "dv12_kt": estado["dv12_kt"],
        "velocidad_kmh": estado["velocidad_kmh"],
        "sin_ang": np.sin(ang),   # + a la derecha del movimiento
        "cos_ang": np.cos(ang),   # + al frente
        "lat_centro": estado["lat"],
        "lon_centro": estado["lon"],
        "elev_celda": oro.z[i, j],
        "tierra_celda": (oro.z[i, j] > 0).astype(float),
        "elev_centro": oro.z[ic, jc],
        "tierra_centro": float(oro.z[ic, jc] > 0),
        "ascenso_orografico": ascenso,
        **{c: contexto.get(c, np.nan) for c in VARIABLES[-4:]},
    })


# ------------------------------------------------------------------ R-CLIPER

def rcliper(vmax_kt, dist_km):
    """Tasa de lluvia (mm/h) del perfil R-CLIPER (coeficientes de Tuleya et al. 2007)."""
    u = 1 + (np.asarray(vmax_kt, dtype=float) - 35) / 33
    t0 = -1.10 + 3.96 * u            # pulgadas/día en el centro
    tm = -1.60 + 4.80 * u            # pulgadas/día en el radio de máxima lluvia
    rm = 64.5 - 13.0 * u             # km
    re = 150.0 - 16.0 * u            # km
    r = np.asarray(dist_km, dtype=float)
    tasa = np.where(r < rm, t0 + (tm - t0) * r / rm, tm * np.exp(-(r - rm) / re))
    return np.clip(tasa, 0, None) * 25.4 / 24


# ------------------------------------------------------------------ conjunto de entrenamiento

def construir(df_ibtracs, caja=config.CAJA_LLUVIA, celdas_por_hora=config.CELDAS_POR_HORA, semilla=11,
              anio_minimo=config.ANIO_MINIMO_LLUVIA):
    """Una fila por (hora de tormenta, celda de CMORPH muestreada) con variables y lluvia observada."""
    rng = np.random.default_rng(semilla)
    oro = Orografia(caja=caja)
    lat_m, lon_m = np.meshgrid(oro.lats, oro.lons, indexing="ij")
    df = df_ibtracs[df_ibtracs["temporada"] >= anio_minimo]
    marcos = []
    for sid, tray in df.groupby("sid"):
        tray = tray.dropna(subset=["vmax_kt"])
        if len(tray) < 2:
            continue
        est = estado_horario(tray)
        est = est[est["lat"].between(caja["lat_min"], caja["lat_max"])
                  & est["lon"].between(caja["lon_min"], caja["lon_max"])
                  & (est["vmax_kt"] >= config.VIENTO_MINIMO_KT) & (est["tiempo"].dt.hour % 6 == 0)]
        if est.empty:
            continue
        observada = lluvia.descargar_horas(est["tiempo"], hilos=8)
        por_tiempo = {pd.Timestamp(e["tiempo"]): e for e in est.to_dict("records")}
        for fila in est.itertuples(index=False):
            t = pd.Timestamp(fila.tiempo)
            campo = observada.get(t)
            # Emisión simulada: un plazo al azar entre los que la tormenta ya existía y había CMORPH.
            plazos = [p for p in PLAZOS_H if (t - pd.Timedelta(hours=int(p))) in por_tiempo
                      and (t - pd.Timedelta(hours=int(p))) in observada]
            if campo is None or not plazos:
                continue
            plazo = int(rng.choice(plazos))
            t0 = t - pd.Timedelta(hours=plazo)
            dist = geo.distancia_km(fila.lat, fila.lon, lat_m, lon_m)
            cerca = np.flatnonzero((dist <= config.RADIO_MAXIMO_LLUVIA_KM) & np.isfinite(campo))
            if cerca.size == 0:
                continue
            # Más muestras cerca del centro (el área crece con la distancia); no sesga E[lluvia | variables].
            peso = 1 / (dist.ravel()[cerca] + 25)
            k = rng.choice(cerca, size=min(celdas_por_hora, cerca.size), replace=False, p=peso / peso.sum())
            lat_k, lon_k = lat_m.ravel()[k], lon_m.ravel()[k]
            ctx = {"plazo_h": plazo, **contexto_emision(observada[t0], por_tiempo[t0], oro, lat_k, lon_k)}
            v = variables(fila._asdict(), lat_k, lon_k, oro, ctx)
            v["lluvia_mm_h"] = campo.ravel()[k]
            v["sid"], v["temporada"], v["tiempo"] = sid, int(tray["temporada"].iloc[0]), fila.tiempo
            marcos.append(v)
    return pd.concat(marcos, ignore_index=True)


# ------------------------------------------------------------------ entrenamiento y evaluación

def entrenar(datos):
    import lightgbm as lgb

    mdl = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.4, n_estimators=600, learning_rate=0.05,
                            num_leaves=63, min_child_samples=100, subsample=0.8, subsample_freq=1,
                            colsample_bytree=0.8, verbose=-1)
    mdl.fit(datos[VARIABLES], datos["lluvia_mm_h"])
    return mdl


def entrenar_severo(datos, alfa=0.9):
    """Escenario severo: el percentil `alfa` de la lluvia horaria (regresión por cuantiles).

    El modelo de valor esperado subestima los eventos de lluvia extrema; este da una cota
    pesimista útil para alertar.
    """
    import lightgbm as lgb

    mdl = lgb.LGBMRegressor(objective="quantile", alpha=alfa, n_estimators=400, learning_rate=0.05, num_leaves=63,
                            min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    mdl.fit(datos[VARIABLES], datos["lluvia_mm_h"])
    return mdl


def predecir(mdl, X):
    return np.clip(mdl.predict(X[VARIABLES]), 0, None)


def evaluar(mdl, prueba):
    """Error por celda-hora y por evento (lluvia acumulada en cada celda muestreada de cada tormenta)."""
    y = prueba["lluvia_mm_h"].to_numpy()
    resultado = {}
    for nombre, pred in (("lightgbm", predecir(mdl, prueba)), ("rcliper", rcliper(prueba["vmax_kt"], prueba["dist_km"]))):
        cerca = prueba["dist_km"].to_numpy() <= 300
        resultado[nombre] = {
            "rmse_mm_h": float(np.sqrt(np.mean((pred - y) ** 2))),
            "mae_mm_h": float(np.mean(np.abs(pred - y))),
            "sesgo_mm_h": float(np.mean(pred - y)),
            "correlacion": float(np.corrcoef(pred, y)[0, 1]),
            "rmse_mm_h_a_300km": float(np.sqrt(np.mean((pred[cerca] - y[cerca]) ** 2))),
            "correlacion_a_300km": float(np.corrcoef(pred[cerca], y[cerca])[0, 1]),
        }
    resultado["filas_prueba"] = int(len(prueba))
    resultado["tormentas_prueba"] = int(prueba["sid"].nunique())
    resultado["lluvia_media_observada_mm_h"] = float(y.mean())
    return resultado


def importancia(mdl):
    gan = mdl.booster_.feature_importance(importance_type="gain")
    return {v: round(float(g / gan.sum()), 4) for v, g in sorted(zip(VARIABLES, gan), key=lambda x: -x[1])}


def guardar(mdl, metricas, carpeta=config.CARPETA_SALIDAS_INUNDACION, nombre="modelo_lluvia"):
    import joblib

    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    joblib.dump(mdl, carpeta / f"{nombre}.joblib")
    if metricas is None:
        return
    (carpeta / "metricas_lluvia.json").write_text(json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8")


def cargar(carpeta=config.CARPETA_SALIDAS_INUNDACION, nombre="modelo_lluvia"):
    import joblib

    return joblib.load(Path(carpeta) / f"{nombre}.joblib")


# ------------------------------------------------------------------ pronóstico en un punto

def serie_en_punto(mdl, trayectoria, lat, lon, emision, oro=None, metodo="lightgbm"):
    """Lluvia horaria pronosticada (mm/h) en (lat, lon) a lo largo de una trayectoria (observada o pronosticada).

    emision: hora (UTC) en que se hace el pronóstico; de ahí salen la lluvia observada de
    contexto (CMORPH) y el plazo de cada hora.
    """
    oro = oro or Orografia()
    est = estado_horario(trayectoria)
    emision = pd.Timestamp(emision).floor("h")
    en_emision = est.iloc[(est["tiempo"] - emision).abs().argmin()]
    campo = lluvia.descargar_hora(emision)
    contexto = contexto_emision(campo, en_emision, oro, lat, lon) if campo is not None else {}
    filas = []
    for e in est.to_dict("records"):
        plazo = float(np.clip((e["tiempo"] - emision) / pd.Timedelta(hours=1), PLAZOS_H[0], PLAZOS_H[-1]))
        filas.append(variables(e, [lat], [lon], oro, {"plazo_h": plazo, **contexto}))
    X = pd.concat(filas, ignore_index=True)
    tasa = predecir(mdl, X) if metodo == "lightgbm" else rcliper(X["vmax_kt"], X["dist_km"])
    return pd.Series(tasa, index=pd.DatetimeIndex(est["tiempo"]), name="lluvia_mm_h")


def evaluar_acumulados(mdl, df_ibtracs, anio_desde=config.ANIO_PRUEBA_LLUVIA, radio_km=300, caja=config.CAJA_LLUVIA,
                       plazo_h=24):
    """Lluvia total por tormenta en cada celda de tierra cercana a la trayectoria: modelo vs CMORPH.

    Es la métrica que importa para inundación: cuánto llovió en todo el evento. La lluvia
    total se aproxima con las horas sinópticas (00, 06, 12, 18 UTC) x 6 h. Cada hora se
    pronostica con lo observado `plazo_h` horas antes (o lo más cercano posible).
    """
    oro = Orografia(caja=caja)
    lat_m, lon_m = np.meshgrid(oro.lats, oro.lons, indexing="ij")
    tierra = oro.z > 0
    filas = []
    for sid, tray in df_ibtracs[df_ibtracs["temporada"] >= anio_desde].groupby("sid"):
        tray = tray.dropna(subset=["vmax_kt"])
        if len(tray) < 2:
            continue
        est = estado_horario(tray)
        est = est[est["lat"].between(caja["lat_min"], caja["lat_max"])
                  & est["lon"].between(caja["lon_min"], caja["lon_max"])
                  & (est["vmax_kt"] >= config.VIENTO_MINIMO_KT) & (est["tiempo"].dt.hour % 6 == 0)]
        if est.empty:
            continue
        observada = lluvia.descargar_horas(est["tiempo"], hilos=8)
        por_tiempo = {pd.Timestamp(e["tiempo"]): e for e in est.to_dict("records")}
        emisiones = sorted(x for x in por_tiempo if x in observada)
        if not emisiones:
            continue
        acum = {k: np.zeros(lat_m.shape) for k in ("obs", "lightgbm", "rcliper")}
        cerca = np.zeros(lat_m.shape, dtype=bool)
        for e in est.to_dict("records"):
            campo = observada.get(pd.Timestamp(e["tiempo"]))
            if campo is None:
                continue
            dist = geo.distancia_km(e["lat"], e["lon"], lat_m, lon_m)
            sel = (dist <= config.RADIO_MAXIMO_LLUVIA_KM) & tierra
            if not sel.any():
                continue
            cerca |= (dist <= radio_km) & tierra
            # Pronóstico emitido `plazo_h` horas antes (o en la primera hora disponible de la tormenta).
            t = pd.Timestamp(e["tiempo"])
            previas = [x for x in emisiones if x <= t - pd.Timedelta(hours=plazo_h)]
            t0 = previas[-1] if previas else emisiones[0]
            plazo = max((t - t0) / pd.Timedelta(hours=1), 6)
            ctx = {"plazo_h": plazo, **contexto_emision(observada[t0], por_tiempo[t0], oro, lat_m[sel], lon_m[sel])}
            X = variables(e, lat_m[sel], lon_m[sel], oro, ctx)
            acum["obs"][sel] += np.nan_to_num(campo[sel]) * 6
            acum["lightgbm"][sel] += predecir(mdl, X) * 6
            acum["rcliper"][sel] += rcliper(X["vmax_kt"], X["dist_km"]) * 6
        if cerca.any():
            filas.append(pd.DataFrame({"sid": sid, "nombre": tray["nombre"].iloc[0], "temporada": tray["temporada"].iloc[0],
                                       "lat": lat_m[cerca], "lon": lon_m[cerca],
                                       **{k: v[cerca] for k, v in acum.items()}}))
    tabla = pd.concat(filas, ignore_index=True)
    metricas = {}
    for k in ("lightgbm", "rcliper"):
        err = tabla[k] - tabla["obs"]
        fuerte = tabla["obs"] >= 100
        metricas[k] = {
            "rmse_mm": float(np.sqrt(np.mean(err ** 2))),
            "sesgo_mm": float(err.mean()),
            "correlacion": float(np.corrcoef(tabla[k], tabla["obs"])[0, 1]),
            "rmse_mm_celdas_100mm_o_mas": float(np.sqrt(np.mean(err[fuerte] ** 2))),
            "sesgo_mm_celdas_100mm_o_mas": float(err[fuerte].mean()),
        }
    metricas["celdas_evento"] = int(len(tabla))
    metricas["tormentas"] = int(tabla["sid"].nunique())
    return metricas, tabla
