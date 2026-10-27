"""Pruebas del módulo de inundación con terreno y lluvia sintéticos (no son resultados reales)."""
import numpy as np
import pandas as pd
import pytest

from inundacion import escenarios, exposicion, hidrologia, modelo_lluvia, terreno


@pytest.fixture(scope="module")
def valle(tmp_path_factory):
    """Valle en V de 3 x 6 km que baja hacia el mar (filas inferiores), en una malla UTM de 30 m."""
    import rasterio
    from rasterio.transform import from_origin

    alto, ancho = 200, 100
    filas, cols = np.mgrid[0:alto, 0:ancho]
    dem = 2.0 + 0.004 * 30 * (alto - filas) + 0.08 * 30 * np.abs(cols - ancho / 2) / 10
    dem[-10:] = 0.0   # mar
    ruta = tmp_path_factory.mktemp("valle") / "dem.tif"
    perfil = {"driver": "GTiff", "height": alto, "width": ancho, "count": 1, "dtype": "float32",
              "crs": terreno.CRS_METRICO, "transform": from_origin(400000, 1860000, 30, 30)}
    with rasterio.open(ruta, "w", **perfil) as dst:
        dst.write(dem.astype("float32"), 1)
    return terreno.procesar(ruta, umbral_cauce_km2=0.2)


def test_mascara_mar_solo_toma_lo_bajo_conectado_al_borde():
    dem = np.full((5, 5), 10.0)
    dem[0, :] = 0
    dem[2, 2] = 0   # depresión interior: no es mar
    mar = terreno.mascara_mar(dem)
    assert mar[0].all() and not mar[2, 2]


def test_terreno_del_valle(valle):
    assert valle["cauce"].sum() > 0
    assert np.nanmin(valle["hand"]) == 0
    # El cauce corre por el centro del valle
    assert abs(np.flatnonzero(valle["cauce"].any(axis=0)).mean() - 50) < 5
    assert valle["area_drenada_km2"].max() > 5


def test_escurrimiento_scs():
    lluvia = np.array([5, 20, 40, 40, 10, 0], dtype=float)
    q = hidrologia.escurrimiento_scs(lluvia, [70, 100])
    assert q[1].sum() == pytest.approx(lluvia.sum())     # CN 100: todo escurre
    assert 0 < q[0].sum() < lluvia.sum()
    assert q[0][0] == 0                                   # la primera lluvia se queda en abstracción inicial
    assert (q >= -1e-9).all()


def test_hidrograma_unitario_conserva_volumen():
    for tc in (0.5, 2.0, 6.0):
        assert hidrologia.hidrograma_unitario(tc).sum() == pytest.approx(1.0)


def test_mas_lluvia_inunda_mas(valle):
    horas = pd.date_range("2024-09-24", periods=36, freq="1h")
    poca = pd.Series(np.where(np.arange(36) < 12, 5.0, 0.0), index=horas)
    mucha = poca * 8
    sim_poca, sim_mucha = hidrologia.simular(valle, poca), hidrologia.simular(valle, mucha)
    area_poca = np.isfinite(hidrologia.tirante_maximo(valle, sim_poca)[0]).sum()
    area_mucha = np.isfinite(hidrologia.tirante_maximo(valle, sim_mucha)[0]).sum()
    assert area_mucha > area_poca
    # El pico del caudal llega después del inicio de la lluvia
    assert sim_mucha["caudal_m3s"].max(axis=0).argmax() > 0


def test_cauce_lleno_conduce_lluvia_ligera(valle):
    ancho, tirante, seccion = hidrologia.cauce_lleno([1, 100])
    assert (np.diff(ancho) > 0).all() and (np.diff(tirante) > 0).all() and (np.diff(seccion) > 0).all()
    horas = pd.date_range("2024-09-24", periods=12, freq="1h")
    sim = hidrologia.simular(valle, pd.Series(np.where(np.arange(12) < 3, 2.0, 0.0), index=horas))
    assert not np.isfinite(hidrologia.tirante_maximo(valle, sim)[0]).any()   # no se desborda


def test_laguna_sube_y_desagua(valle):
    ter = dict(valle)
    agua = np.zeros(ter["dem"].shape, dtype=bool)
    agua[175:188, 35:65] = True       # laguna al pie del valle, junto al mar
    ter["agua_permanente"] = agua
    lat, lon = exposicion.a_lat_lon(ter, [181], [50])
    horas = pd.date_range("2024-09-24", periods=96, freq="1h")
    lluvia = pd.Series(np.where(np.arange(96) < 12, 40.0, 0.0), index=horas)
    sim = hidrologia.simular(ter, lluvia, lagunas={"Prueba": (lat[0], lon[0])})
    nivel = sim["lagunas"][0]["nivel_m"]
    assert nivel.max() > 0.2 and nivel[-1] < nivel.max() / 2      # sube con la tormenta y luego baja
    assert sim["lagunas"][0]["entrada_m3s"].max() > 0              # el río del valle le entra
    orilla = sim["lagunas"][0]["orilla"]
    pico = int(nivel.argmax())
    assert np.isfinite(hidrologia.tirante_en_celdas(ter, sim, pico).ravel()[orilla]).any()


def test_exposicion_de_inmuebles(valle):
    horas = pd.date_range("2024-09-24", periods=24, freq="1h")
    sim = hidrologia.simular(valle, pd.Series(60.0, index=horas))
    cartera = exposicion.cartera_simulada(valle, n=40, elevacion_maxima=1000)
    res, tirante = exposicion.exposicion(valle, sim, cartera)
    assert tirante.shape == (40, 24)
    assert set(res["nivel"]) <= {"Alto", "Medio", "Sin afectación"}
    assert (res["tirante_max_m"] >= 0).all()


def test_rcliper():
    assert modelo_lluvia.rcliper(100, 50) > modelo_lluvia.rcliper(100, 300)
    assert modelo_lluvia.rcliper(120, 50) > modelo_lluvia.rcliper(40, 50)


def test_variables_lado_derecho_y_frente():
    oro = modelo_lluvia.Orografia(elevacion=np.zeros((92, 144)))
    estado = {"lat": 16.0, "lon": -100.0, "rumbo": 0.0, "vmax_kt": 80, "dv12_kt": 0, "velocidad_kmh": 15}
    v = modelo_lluvia.variables(estado, [16.0, 16.0, 17.0], [-99.0, -101.0, -100.0], oro)
    # Moviéndose al norte: el este queda a la derecha, el oeste a la izquierda y el norte al frente
    assert v["sin_ang"].iloc[0] > 0.9 and v["sin_ang"].iloc[1] < -0.9 and v["cos_ang"].iloc[2] > 0.9
    assert list(v.columns) == modelo_lluvia.VARIABLES


def test_fecha_de_emision_nhc():
    assert escenarios.emision_nhc("1200 PM CST Mon Sep 23 2024") == pd.Timestamp("2024-09-23 18:00")
    assert escenarios.emision_nhc("900 PM CST Mon Sep 23 2024") == pd.Timestamp("2024-09-24 03:00")
    assert escenarios.emision_nhc("300 AM CST Mon Sep 23 2024") == pd.Timestamp("2024-09-23 09:00")
