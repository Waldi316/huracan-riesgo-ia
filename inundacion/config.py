"""Parámetros del módulo de inundación. Todo lo ajustable vive aquí."""
from huracan_ia.config import CARPETA_DATOS, CARPETA_SALIDAS

CARPETA_DATOS_INUNDACION = CARPETA_DATOS / "inundacion"
CARPETA_SALIDAS_INUNDACION = CARPETA_SALIDAS / "inundacion"

# --- Fuentes públicas ---
# Copernicus DEM GLO-30 (30 m), servido como COG en AWS Open Data.
URL_DEM_COPERNICUS = (
    "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM/"
    "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM.tif"
)
# ESA WorldCover 2021 (10 m), cobertura de suelo para el número de curva.
URL_WORLDCOVER = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
    "ESA_WorldCover_10m_2021_v200_{ns}{lat:02d}{ew}{lon:03d}_Map.tif"
)
# CMORPH CDR (NOAA/NCEI): lluvia estimada por satélite, horaria a 0.25°, desde 1998.
URL_CMORPH_HORARIO = (
    "https://www.ncei.noaa.gov/data/cmorph-high-resolution-global-precipitation-estimates/access/hourly/0.25deg/"
    "{t:%Y/%m/%d}/CMORPH_V1.0_ADJ_0.25deg-HLY_{t:%Y%m%d%H}.nc"
)
# Elevación gruesa (teselas Terrarium/Mapzen en AWS) para el efecto orográfico en el modelo de lluvia.
URL_ELEVACION_TESELA = "https://s3.amazonaws.com/elevation-tiles-prod/geotiff/{z}/{x}/{y}.tif"
NIVEL_TESELA_ELEVACION = 5

# --- Zona piloto ---
DOMINIOS = {
    # Acapulco: bahía, río La Sabana y llanuras de Llano Largo / Diamante.
    # Las lagunas (un punto dentro de cada una) se simulan como vasos que se llenan y desaguan.
    "acapulco": {"lat_min": 16.74, "lat_max": 17.06, "lon_min": -99.99, "lon_max": -99.70,
                 "lagunas": {"Tres Palos": (16.80, -99.75)}},
    # Manzanillo: puerto, laguna de Las Garzas / San Pedrito y el vaso I de Cuyutlán (Trudy 2026).
    "manzanillo": {"lat_min": 18.95, "lat_max": 19.20, "lon_min": -104.45, "lon_max": -104.15,
                   "lagunas": {"Cuyutlán vaso I": (19.033, -104.261, 8), "Cuyutlán vaso II": (19.0015, -104.18, 5),
                               "Las Garzas": (19.0677, -104.2932, 5)}},
    # Polo 2026 (Guaymas, Ciudad Constitución) y Rachel 2026 (Tijuana, Ensenada, Mexicali).
    "guaymas": {"lat_min": 27.85, "lat_max": 28.05, "lon_min": -111.05, "lon_max": -110.70},
    "constitucion": {"lat_min": 24.95, "lat_max": 25.12, "lon_min": -111.78, "lon_max": -111.55},
    "tijuana": {"lat_min": 32.40, "lat_max": 32.56, "lon_min": -117.13, "lon_max": -116.85},
    "ensenada": {"lat_min": 31.78, "lat_max": 31.92, "lon_min": -116.68, "lon_max": -116.54},
    "mexicali": {"lat_min": 32.53, "lat_max": 32.68, "lon_min": -115.58, "lon_max": -115.35},
}

# --- Modelo de lluvia ---
# Caja en la que se toman tormentas y celdas de CMORPH para entrenar.
CAJA_LLUVIA = {"lat_min": 10.0, "lat_max": 33.0, "lon_min": -118.0, "lon_max": -82.0}
ANIO_MINIMO_LLUVIA = 2004
RADIO_MAXIMO_LLUVIA_KM = 600
CELDAS_POR_HORA = 400          # celdas muestreadas al azar por cada hora de tormenta
VIENTO_MINIMO_KT = 25
ANIO_PRUEBA_LLUVIA = 2021      # se entrena con temporadas anteriores y se evalúa con las siguientes

# --- Hidrología ---
UMBRAL_CAUCE_KM2 = 0.5         # área drenada a partir de la cual una celda es cauce
N_MANNING = 0.06               # rugosidad de planicie (urbano / vegetación)
N_MANNING_CAUCE = 0.035        # rugosidad del cauce principal
PENDIENTE_MINIMA = 0.001
NIVELES_TIRANTE_M = (0.0, 20.0, 0.1)   # tabla de curva de gasto: de 0 a 20 m cada 10 cm
TIRANTE_MINIMO_VISIBLE_M = 0.15
# Laguna costera como vaso: sube con lo que le entra y desagua al mar por un vertedor en la barra.
# El ancho de salida es el parámetro de calibración (validado con Sentinel-1 / CENAPRED, John 2024).
LAGUNA_ANCHO_SALIDA_M = 100.0   # CSI 0.40 contra Sentinel-1 del 28 sep 2024 (John)
LAGUNA_COEF_VERTEDOR = 1.7     # vertedor de cresta ancha, m^0.5/s
LAGUNA_SOBREELEVACION_MAX_M = 8.0
# Número de curva (condición de humedad II) por clase de WorldCover.
NUMERO_CURVA = {
    10: 70,   # árboles
    20: 77,   # matorral
    30: 79,   # pastizal
    40: 81,   # cultivo
    50: 92,   # urbano
    60: 86,   # suelo desnudo
    70: 98,   # nieve
    80: 100,  # agua
    90: 90,   # humedal
    95: 90,   # manglar
    100: 79,  # musgo
}
NUMERO_CURVA_DEFECTO = 80
# Con huracanes el suelo suele llegar mojado: se usa la condición III.
HUMEDAD_ANTECEDENTE_III = True
