"""Parámetros del proyecto. Todo lo ajustable vive aquí."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
CARPETA_DATOS = RAIZ / "datos"
CARPETA_SALIDAS = RAIZ / "salidas"

# --- Fuentes públicas ---
URL_IBTRACS = (
    "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/"
    "v04r01/access/csv/ibtracs.{cuenca}.list.v04r01.csv"
)
CUENCAS_IBTRACS = ("NA", "EP")  # Atlántico Norte y Pacífico Oriental
URL_NHC_TORMENTAS_ACTIVAS = "https://www.nhc.noaa.gov/CurrentStorms.json"

# Los radios de viento por cuadrante de IBTrACS (USA_R34_*) existen desde 2004.
ANIO_MINIMO_HISTORICO = 2004

# --- Ventana de pronóstico ---
HORIZONTE_HORAS = 72
PASO_PRONOSTICO_HORAS = 6
PASO_EMISION_HORAS = 6

# Radio del cono del NHC (millas náuticas) por horas de pronóstico.
# Valores aproximados del cono atlántico de temporadas recientes; el NHC los
# actualiza cada año en https://www.nhc.noaa.gov/aboutcone.shtml
RADIO_CONO_MN = {0: 0, 12: 26, 24: 39, 36: 52, 48: 66, 60: 81, 72: 99, 96: 145, 120: 205}

# Solo se evalúan ubicaciones a menos de esta distancia del centro actual.
DISTANCIA_MAXIMA_EVALUACION_KM = 1200

# Caja aproximada de México para elegir tormentas relevantes.
CAJA_MEXICO = {"lat_min": 10.0, "lat_max": 35.0, "lon_min": -120.0, "lon_max": -84.0}

# --- Índice base (método de pesos fijos, como el proceso operativo actual) ---
PESOS_AMENAZA = {"ubicacion_cono": 0.15, "viento": 0.30, "dist_pronostico": 0.30, "dist_observado": 0.25}
PESO_AMENAZA = 0.5
PESO_VULNERABILIDAD = 0.5
CORTE_BAJO = 0.4
CORTE_ALTO = 0.6
BUFFER_TRAYECTORIA_KM = 500
TABLA_VIENTO_KT = [(0, 41, 0.2), (41.1, 74, 0.4), (74.1, 107, 0.6), (107.1, 140, 0.8), (140.1, float("inf"), 1.0)]
TABLA_DIST_PRONOSTICO_KM = [(76, 1.0), (176, 0.8), (276, 0.6), (376, 0.4)]
TABLA_DIST_OBSERVADO_KM = [(139, 1.0), (289, 0.8), (398, 0.6), (468, 0.4)]

# --- Niveles de alerta a partir de la probabilidad del modelo ---
UMBRAL_PROB_ALTO = 0.6
UMBRAL_PROB_MEDIO = 0.3

# --- Boletín con LLM ---
MODELO_CLAUDE = "claude-opus-5-5"
MAXIMO_UBICACIONES_EN_BOLETIN = 15
