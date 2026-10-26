"""Línea de comandos del módulo de inundación.

    python -m inundacion terreno --dominio acapulco
    python -m inundacion entrenar-lluvia
    python -m inundacion simular --dominio acapulco --tormenta JOHN --temporada 2024 \\
        --inicio "2024-09-22 12:00" --fin "2024-09-29 12:00" --nhc datos/inundacion/nhc/ep102024_5day_005.zip
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from huracan_ia import ibtracs

from . import animacion, config, escenarios, exportar, exposicion, hidrologia, modelo_lluvia, terreno


def _terreno(nombre):
    ruta = config.CARPETA_DATOS_INUNDACION / f"{nombre}_terreno.npz"
    if not ruta.exists():
        ruta_dem, ruta_cob = terreno.descargar(nombre)
        print("Calculando drenaje y HAND...")
        terreno.procesar(ruta_dem, ruta_cob, ruta_salida=ruta)
    return terreno.cargar(ruta)


def cmd_terreno(args):
    ter = _terreno(args.dominio)
    print(f"Malla {ter['dem'].shape}, celdas de cauce: {ter['cauce'].sum():,}, "
          f"área drenada máxima: {ter['area_drenada_km2'].max():.0f} km²")


def cmd_entrenar_lluvia(args):
    df = ibtracs.cargar(ibtracs.descargar())
    print("Construyendo conjunto de entrenamiento (IBTrACS x CMORPH)...")
    datos = modelo_lluvia.construir(df)
    entrenamiento = datos[datos["temporada"] < args.anio_prueba]
    prueba = datos[datos["temporada"] >= args.anio_prueba]
    print(f"Entrenamiento: {len(entrenamiento):,} filas · Prueba: {len(prueba):,} (temporadas >= {args.anio_prueba})")
    mdl = modelo_lluvia.entrenar(entrenamiento)
    metricas = modelo_lluvia.evaluar(mdl, prueba)
    metricas["importancia_variables"] = modelo_lluvia.importancia(mdl)
    metricas["acumulado_por_evento"], _ = modelo_lluvia.evaluar_acumulados(mdl, df, anio_desde=args.anio_prueba)
    print("Entrenando el escenario severo (percentil 90)...")
    severo = modelo_lluvia.entrenar_severo(entrenamiento)
    metricas["severo_p90"] = {
        "cobertura_horaria": float((prueba["lluvia_mm_h"] <= modelo_lluvia.predecir(severo, prueba)).mean()),
        "acumulado_por_evento": modelo_lluvia.evaluar_acumulados(severo, df, anio_desde=args.anio_prueba)[0]["lightgbm"],
    }
    modelo_lluvia.guardar(mdl, metricas)
    modelo_lluvia.guardar(severo, None, nombre="modelo_lluvia_p90")
    print(json.dumps(metricas, indent=2, ensure_ascii=False))


def cmd_simular(args):
    dominio = config.DOMINIOS[args.dominio]
    ter = _terreno(args.dominio)
    mdl = modelo_lluvia.cargar()
    severo = modelo_lluvia.cargar(nombre="modelo_lluvia_p90")
    horas = escenarios.ventana(args.inicio, args.fin)

    df = ibtracs.cargar(ibtracs.descargar())
    sid = ibtracs.buscar_tormenta(df, args.tormenta, args.temporada)
    real = df[df["sid"] == sid].dropna(subset=["vmax_kt"])

    series = {"Lluvia observada (CMORPH)": escenarios.observada(dominio, horas)}
    trayectorias = [{"nombre": f"{args.tormenta.title()} (trayectoria real)", "lat": real["lat"].tolist(),
                     "lon": real["lon"].tolist()}]
    for ruta in args.nhc or []:
        emision, tray = escenarios.pronostico_nhc(ruta)
        etiqueta = f"Pronóstico NHC {emision.tz_localize('UTC').tz_convert(animacion.ZONA_HORARIA):%d %b %H:%M}"
        series[etiqueta] = escenarios.con_trayectoria(mdl, tray, dominio, horas, emision=emision)
        series[etiqueta + " · severo P90"] = escenarios.con_trayectoria(severo, tray, dominio, horas, emision=emision)
        series[etiqueta + " · R-CLIPER"] = escenarios.con_trayectoria(mdl, tray, dominio, horas, emision=emision,
                                                                      metodo="rcliper")
        trayectorias.append({"nombre": etiqueta, "lat": tray["lat"].tolist(), "lon": tray["lon"].tolist()})
    if args.emision_real:
        series["Trayectoria real + modelo"] = escenarios.con_trayectoria(mdl, real, dominio, horas,
                                                                         emision=pd.Timestamp(args.emision_real))

    cartera = exposicion.cartera_simulada(ter, n=args.n_inmuebles)
    carpeta = Path(args.salida) / f"{args.tormenta.lower()}{args.temporada}_{args.dominio}"
    carpeta.mkdir(parents=True, exist_ok=True)
    resumen, lista = {}, []
    for i, (nombre, serie) in enumerate(series.items()):
        sim = hidrologia.simular(ter, serie)
        res, tirante = exposicion.exposicion(ter, sim, cartera)
        area = exposicion.area_inundada_km2(ter, sim)
        resumen[nombre] = {"lluvia_total_mm": round(escenarios.lluvia_total_mm(serie), 1),
                           "lluvia_max_mm_h": round(float(serie.max()), 1),
                           "area_inundada_max_km2": round(float(area.max()), 2),
                           "hora_pico": str(pd.Timestamp(sim["horas"][int(area.argmax())])),
                           "inmuebles": exposicion.resumen(res)}
        print(f"{nombre}: {json.dumps(resumen[nombre], ensure_ascii=False)}")
        base = f"escenario{i}"
        exportar.todo(ter, sim, res, carpeta, base)
        lista.append({"nombre": nombre, "sim": sim, "res": res, "tirante_inmuebles": tirante})
    (carpeta / "resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame({k: v.to_numpy() for k, v in series.items()}, index=horas).to_csv(carpeta / "lluvia_horaria.csv")

    # La animación lleva la lluvia observada, el último aviso del NHC (esperado y severo) y la trayectoria real.
    ultimo = trayectorias[-1]["nombre"] if args.nhc else None
    visibles = [e for e in lista if e["nombre"].startswith("Lluvia observada") or e["nombre"].startswith("Trayectoria real")
                or (ultimo and e["nombre"].startswith(ultimo) and "R-CLIPER" not in e["nombre"])]
    horas_anim = list(exportar.horas_con_agua(lista[0]["sim"]))[::args.paso_animacion]
    ruta = animacion.generar(
        ter, visibles, carpeta / "animacion.html", horas=horas_anim,
        titulo=f"{args.tormenta.title()} {args.temporada} · inundación simulada en {args.dominio.title()}",
        trayectorias=trayectorias,
        nota="Modelo de primer orden (HAND + curva de gasto) con DEM Copernicus 30 m. Inmuebles simulados.",
    )
    print(f"Salidas en {carpeta}/ · animación: {ruta}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="inundacion", description="Zonas inundables por lluvia de huracanes")
    p.add_argument("--salida", default=str(config.CARPETA_SALIDAS_INUNDACION))
    sub = p.add_subparsers(dest="comando", required=True)

    sp = sub.add_parser("terreno", help="Descarga el DEM y calcula drenaje y HAND")
    sp.add_argument("--dominio", default="acapulco", choices=sorted(config.DOMINIOS))
    sp.set_defaults(func=cmd_terreno)

    sp = sub.add_parser("entrenar-lluvia", help="Entrena el modelo de lluvia (LightGBM) y lo compara con R-CLIPER")
    sp.add_argument("--anio-prueba", type=int, default=config.ANIO_PRUEBA_LLUVIA)
    sp.set_defaults(func=cmd_entrenar_lluvia)

    sp = sub.add_parser("simular", help="Simula la inundación de una tormenta histórica con varios escenarios")
    sp.add_argument("--dominio", default="acapulco", choices=sorted(config.DOMINIOS))
    sp.add_argument("--tormenta", required=True)
    sp.add_argument("--temporada", required=True, type=int)
    sp.add_argument("--inicio", required=True, help="UTC")
    sp.add_argument("--fin", required=True, help="UTC")
    sp.add_argument("--nhc", action="append", help="Zip '5day' de un aviso del NHC (se puede repetir)")
    sp.add_argument("--emision-real", help="Agrega el escenario con la trayectoria real emitido a esta hora (UTC)")
    sp.add_argument("--n-inmuebles", type=int, default=250)
    sp.add_argument("--paso-animacion", type=int, default=2, help="Horas entre cuadros de la animación")
    sp.set_defaults(func=cmd_simular)

    args = p.parse_args(argv)
    args.func(args)
