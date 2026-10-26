"""Línea de comandos.

    python -m huracan_ia descargar
    python -m huracan_ia entrenar [--ubicaciones mis_ubicaciones.csv] [--anio-prueba 2020]
    python -m huracan_ia simular --tormenta OTIS --temporada 2023 --emision "2023-10-24 00:00" [--boletin]
    python -m huracan_ia alertar --activas | --zip al182023_5day_001.zip [--boletin]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import boletin, config, dataset, etiquetas, ibtracs, indice_base, mapa, modelo, pronostico, ubicaciones, variables


def _cartera(args):
    if args.ubicaciones:
        return ubicaciones.desde_csv(args.ubicaciones)
    print("Usando cartera SIMULADA (sin datos de ninguna empresa).")
    return ubicaciones.simuladas(n=args.n_simuladas)


def evaluar_tormenta(estado, pron, cartera, mdl, meta):
    """Probabilidad del modelo + índice base para cada ubicación de la cartera."""
    if np.isnan(estado.get("r34_max_km", np.nan)):
        estado = {**estado, "r34_max_km": meta["r34_mediana_km"]}
    v = variables.calcular(estado, pron, cartera)
    v = indice_base.calcular(v)
    v["prob"] = modelo.predecir(mdl, v)
    v["nivel"] = modelo.nivel(v["prob"])
    info = cartera.rename(columns={"lat": "lat_ubicacion", "lon": "lon_ubicacion"})
    return info.merge(v, on="id")


def _guardar_salidas(nombre, emision, estado, pron, resultados, args, trayectoria_real=None):
    carpeta = Path(args.salida)
    carpeta.mkdir(parents=True, exist_ok=True)
    base = f"{nombre}_{pd.Timestamp(emision):%Y%m%d_%H%M}".replace(" ", "_")
    columnas = ["id", "nombre", "ciudad", "estado", "lat_ubicacion", "lon_ubicacion", "prob", "nivel",
                "ifa", "riesgo_base", "nivel_base", "dist_min_km", "horas_a_minima"]
    columnas += [c for c in ("viento_recibido_kt",) if c in resultados.columns]
    resultados[columnas].sort_values("prob", ascending=False).round(4).to_csv(carpeta / f"{base}_alertas.csv", index=False)
    mapa.generar(resultados, pron, f"{nombre} · {emision}", carpeta / f"{base}_mapa.html", trayectoria_real)
    print(f"Salidas en {carpeta}/{base}_*")
    print(resultados["nivel"].value_counts().reindex(["Alto", "Medio", "Bajo"], fill_value=0).to_string())
    if args.boletin:
        texto = boletin.generar(boletin.resumen_para_boletin(nombre, emision, estado, resultados))
        (carpeta / f"{base}_boletin.md").write_text(texto, encoding="utf-8")
        print(f"Boletín: {carpeta}/{base}_boletin.md")


def cmd_descargar(args):
    for ruta in ibtracs.descargar():
        print("OK", ruta)


def cmd_entrenar(args):
    df = ibtracs.cargar(ibtracs.descargar())
    cartera = _cartera(args)
    print(f"Construyendo conjunto de entrenamiento con {len(cartera)} ubicaciones...")
    datos = dataset.construir(df, cartera)
    entrenamiento, prueba = modelo.separar_por_temporada(datos, args.anio_prueba)
    print(f"Entrenamiento: {len(entrenamiento):,} filas · Prueba: {len(prueba):,} filas (temporadas >= {args.anio_prueba})")
    mdl = modelo.entrenar(entrenamiento)
    metricas = modelo.evaluar(mdl, prueba)
    metricas["importancia_variables"] = modelo.importancia(mdl)
    modelo.guardar(mdl, metricas, entrenamiento, carpeta=args.salida)
    print(json.dumps(metricas, indent=2, ensure_ascii=False))


def cmd_simular(args):
    """Backtest de una emisión histórica: qué habría dicho el sistema y qué pasó en realidad."""
    mdl, meta = modelo.cargar(Path(args.salida) / "modelo.joblib")
    df = ibtracs.cargar(ibtracs.descargar())
    sid = ibtracs.buscar_tormenta(df, args.tormenta, args.temporada)
    tray = ibtracs.interpolar_horario(df[df["sid"] == sid])
    t0 = pd.Timestamp(args.emision)
    resultado = pronostico.por_persistencia(tray, t0)
    if resultado is None:
        raise SystemExit(f"No hay datos de {args.tormenta} en {t0} (o 12 h antes)")
    estado, pron = resultado
    cartera = _cartera(args)
    res = evaluar_tormenta(estado, pron, cartera, mdl, meta)
    res["viento_recibido_kt"] = etiquetas.viento_maximo_recibido(
        tray, res["lat_ubicacion"], res["lon_ubicacion"],
        t_inicio=t0, t_fin=t0 + pd.Timedelta(hours=config.HORIZONTE_HORAS),
    )
    real = tray[(tray["tiempo"] >= t0 - pd.Timedelta(hours=24)) & (tray["tiempo"] <= t0 + pd.Timedelta(hours=96))]
    _guardar_salidas(args.tormenta.upper(), t0, estado, pron, res, args, trayectoria_real=real)


def cmd_alertar(args):
    mdl, meta = modelo.cargar(Path(args.salida) / "modelo.joblib")
    cartera = _cartera(args)
    if args.zip:
        fuentes = [(args.zip, None)]
    else:
        activas = pronostico.tormentas_activas_nhc()
        if not activas:
            print("El NHC no reporta tormentas activas.")
            return
        fuentes = [(pronostico.descargar_zip(s["forecastTrack"]["zipFile"]), s.get("name")) for s in activas
                   if s.get("forecastTrack", {}).get("zipFile")]
    for fuente, nombre_json in fuentes:
        nombre, estado, pron = pronostico.desde_nhc(fuente, r34_max_km=args.r34_km if args.r34_km else np.nan)
        res = evaluar_tormenta(estado, pron, cartera, mdl, meta)
        _guardar_salidas(nombre_json or nombre, pd.Timestamp.now(tz="America/Mexico_City").tz_localize(None).floor("min"),
                         estado, pron, res, args)


def main(argv=None):
    p = argparse.ArgumentParser(prog="huracan_ia", description="Riesgo de huracanes por ubicación con ML + LLM")
    p.add_argument("--salida", default=str(config.CARPETA_SALIDAS))
    sub = p.add_subparsers(dest="comando", required=True)

    def con_cartera(sp):
        sp.add_argument("--ubicaciones", help="CSV con id, lat, lon (o el CSV del DENUE)")
        sp.add_argument("--n-simuladas", type=int, default=400)
        return sp

    sub.add_parser("descargar", help="Descarga IBTrACS").set_defaults(func=cmd_descargar)

    sp = con_cartera(sub.add_parser("entrenar", help="Entrena y evalúa el modelo contra el índice base"))
    sp.add_argument("--anio-prueba", type=int, default=2020)
    sp.set_defaults(func=cmd_entrenar)

    sp = con_cartera(sub.add_parser("simular", help="Backtest de una emisión histórica"))
    sp.add_argument("--tormenta", required=True)
    sp.add_argument("--temporada", required=True, type=int)
    sp.add_argument("--emision", required=True, help='Hora UTC, p. ej. "2023-10-24 00:00"')
    sp.add_argument("--boletin", action="store_true")
    sp.set_defaults(func=cmd_simular)

    sp = con_cartera(sub.add_parser("alertar", help="Alerta con el pronóstico vigente del NHC"))
    grupo = sp.add_mutually_exclusive_group(required=True)
    grupo.add_argument("--activas", action="store_true")
    grupo.add_argument("--zip", help="Zip del pronóstico a 5 días del NHC")
    sp.add_argument("--r34-km", type=float, help="Radio máximo de 34 kt actual, si se conoce")
    sp.add_argument("--boletin", action="store_true")
    sp.set_defaults(func=cmd_alertar)

    args = p.parse_args(argv)
    args.func(args)
