"""Pruebas con tormentas sintéticas (no son resultados del modelo real)."""
import numpy as np
import pandas as pd
import pytest

from huracan_ia import dataset, etiquetas, geo, ibtracs, indice_base, modelo, pronostico, ubicaciones, variables


def tormenta_sintetica(sid="SINT1", lat=18.0, lon_inicio=-85.0, km_por_hora=20.0, horas=96, r34_km=150.0,
                       r64_km=40.0, vmax=90.0, temporada=2015, rumbo=270.0, tiempo0="2015-09-01 00:00"):
    t = pd.date_range(tiempo0, periods=horas // 6 + 1, freq="6h")
    pasos = np.arange(len(t)) * 6 * km_por_hora
    lats, lons = geo.destino(lat, lon_inicio, rumbo, pasos)
    df = pd.DataFrame({"sid": sid, "temporada": temporada, "nombre": sid, "tiempo": t, "lat": lats, "lon": lons,
                       "vmax_kt": vmax})
    for c in ("ne", "se", "sw", "nw"):
        df[f"r34_{c}_km"] = r34_km
        df[f"r50_{c}_km"] = (r34_km + r64_km) / 2
        df[f"r64_{c}_km"] = r64_km
    return df


def test_distancia_un_grado_en_el_ecuador():
    assert geo.distancia_km(0, 0, 0, 1) == pytest.approx(111.19, abs=0.05)


def test_distancia_este_oeste_menor_que_grados_por_111():
    # A 20°N un grado de longitud mide ~6% menos que 111.32 km
    assert geo.distancia_km(20, -100, 20, -99) == pytest.approx(104.5, abs=0.5)


def test_destino_y_rumbo_son_consistentes():
    lat, lon = geo.destino(18.0, -90.0, 45.0, 300.0)
    assert geo.distancia_km(18.0, -90.0, lat, lon) == pytest.approx(300.0, rel=1e-6)
    assert geo.rumbo_deg(18.0, -90.0, lat, lon) == pytest.approx(45.0, abs=0.01)


def test_cuadrantes():
    assert list(geo.cuadrante([10, 100, 200, 300])) == [0, 1, 2, 3]


def test_etiquetas_por_radio_de_viento():
    tray = ibtracs.interpolar_horario(tormenta_sintetica())
    lat_c, lon_c = 18.0, -92.0  # sobre la trayectoria
    lat_100n, _ = geo.destino(lat_c, lon_c, 0, 100)
    lat_400n, _ = geo.destino(lat_c, lon_c, 0, 400)
    viento = etiquetas.viento_maximo_recibido(tray, [lat_c, lat_100n, lat_400n], [lon_c, lon_c, lon_c])
    assert list(viento) == [64, 34, 0]


def test_etiquetas_respetan_la_ventana():
    tray = ibtracs.interpolar_horario(tormenta_sintetica())
    t_fin = tray["tiempo"].iloc[0] + pd.Timedelta(hours=6)
    viento = etiquetas.viento_maximo_recibido(tray, [18.0], [-99.0], t_fin=t_fin)
    assert viento[0] == 0


def test_persistencia_sigue_trayectoria_recta():
    tray = ibtracs.interpolar_horario(tormenta_sintetica())
    t0 = tray["tiempo"].iloc[24]
    estado, pron = pronostico.por_persistencia(tray, t0)
    assert estado["velocidad_kmh"] == pytest.approx(20.0, rel=0.02)
    real = tray.set_index("tiempo").loc[t0 + pd.Timedelta(hours=48)]
    p48 = pron[pron["horas"] == 48].iloc[0]
    assert geo.distancia_km(real["lat"], real["lon"], p48["lat"], p48["lon"]) < 15


def test_lado_derecho_de_la_trayectoria():
    tray = ibtracs.interpolar_horario(tormenta_sintetica())  # avanza al oeste
    t0 = tray["tiempo"].iloc[12]
    estado, pron = pronostico.por_persistencia(tray, t0)
    ubic = pd.DataFrame({"id": ["norte", "sur"], "lat": [19.0, 17.0], "lon": [-92.0, -92.0]})
    v = variables.calcular(estado, pron, ubic)
    # Avanzando al oeste, el norte queda a la derecha
    assert v.loc[v["id"] == "norte", "lado_derecho"].iloc[0] > 0.9
    assert v.loc[v["id"] == "sur", "lado_derecho"].iloc[0] < -0.9


def test_indice_base_escalones():
    v = pd.DataFrame({
        "id": ["a", "b"], "margen_cono_km": [-10.0, 600.0], "dist_min_km": [50.0, 900.0],
        "dist_actual_km": [100.0, 900.0], "vmax_en_minima_kt": [120.0, 120.0],
    })
    r = indice_base.calcular(v)
    # a: cono 1.0*0.15 + viento 0.8*0.30 + dist 1.0*0.30 + obs 1.0*0.25
    assert r["ifa"].iloc[0] == pytest.approx(0.94)
    assert r["nivel_base"].iloc[0] == "Alto"
    assert r["ifa"].iloc[1] == 0 and r["nivel_base"].iloc[1] == "Sin afectación"


def test_pipeline_completo_con_tormentas_sinteticas():
    rng = np.random.default_rng(0)
    tormentas = []
    for i in range(16):
        tormentas.append(tormenta_sintetica(
            sid=f"S{i}", lat=rng.uniform(15, 22), lon_inicio=rng.uniform(-88, -84), km_por_hora=rng.uniform(12, 25),
            r34_km=rng.uniform(80, 250), rumbo=rng.uniform(250, 300), temporada=2010 + i,
            tiempo0=f"{2010 + i}-09-01 00:00",
        ))
    df = pd.concat(tormentas, ignore_index=True)
    cartera = ubicaciones.simuladas(n=120)
    datos = dataset.construir(df, cartera, sids=sorted(df["sid"].unique()), verbose=False)
    assert datos["y34"].sum() > 0
    entrenamiento, prueba = modelo.separar_por_temporada(datos, 2022)
    mdl = modelo.entrenar(entrenamiento)
    metricas = modelo.evaluar(mdl, prueba)
    assert 0 <= metricas["modelo"]["roc_auc"] <= 1
    assert set(modelo.nivel([0.1, 0.4, 0.9])) == {"Bajo", "Medio", "Alto"}


def test_cartera_simulada_es_reproducible():
    a, b = ubicaciones.simuladas(50, semilla=1), ubicaciones.simuladas(50, semilla=1)
    pd.testing.assert_frame_equal(a, b)
    assert a["id"].is_unique
