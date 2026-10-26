# Riesgo de huracanes por ubicación con IA

Sistema que estima, para cada inmueble de una cartera, la **probabilidad de recibir vientos de tormenta tropical (34 kt o más) en las próximas 72 horas**, a partir del pronóstico del Centro Nacional de Huracanes (NHC). Compara esa probabilidad contra un índice de riesgo de pesos fijos, como el que usan muchos procesos operativos de alertamiento, y redacta un boletín ejecutivo con un LLM (Claude).

Todo funciona con **datos públicos**. Las ubicaciones pueden venir de un CSV propio, del DENUE de INEGI o de una cartera simulada incluida en el proyecto.

## Por qué

Los procesos de alertamiento suelen cruzar el cono del pronóstico con las ubicaciones y asignar un riesgo con reglas fijas: escalones de distancia (menos de 76 km, menos de 176 km…), escalones de viento y pesos definidos a criterio. Funciona, pero:

- Los pesos y cortes nunca se validan contra lo que realmente pasó.
- Los escalones tratan igual una ubicación a 77 km que una a 175 km.
- No distinguen el lado derecho de la trayectoria (el más peligroso en el hemisferio norte) ni el tamaño de la tormenta.

Este proyecto aprende esa relación de huracanes históricos y mide si mejora al índice fijo.

## Cómo funciona

```mermaid
flowchart LR
    A[IBTrACS<br/>trayectorias y radios de viento 2004+] --> B[Emisiones históricas cada 6 h]
    B --> C[Pronóstico por persistencia<br/>lo que se sabía en ese momento]
    C --> D[Variables por ubicación]
    A --> E[Etiqueta: ¿recibió 34 kt en las<br/>72 h siguientes? radios por cuadrante]
    D --> F[LightGBM]
    E --> F
    D --> G[Índice de pesos fijos<br/>línea base]
    F --> H[Backtest por temporada<br/>modelo vs. índice]
    G --> H
    I[NHC: pronóstico vigente] --> D2[Variables] --> F2[Probabilidad y nivel] --> J[Mapa + CSV + boletín con Claude]
```

**Etiquetas sin datos privados.** La mejor trayectoria de IBTrACS trae, cada 6 horas, el radio de vientos de 34, 50 y 64 kt en cada cuadrante (NE, SE, SW, NW). Una ubicación "recibió" 34 kt si en alguna hora quedó dentro del radio de su cuadrante. Eso permite entrenar y evaluar con lo que realmente ocurrió.

**Variables por ubicación** (`huracan_ia/variables.py`): distancia actual al centro, distancia mínima al pronóstico y en cuántas horas ocurre, viento actual y pronosticado en ese punto, lado de la trayectoria (−1 izquierda, +1 derecha), margen respecto al cono del NHC, radio de 34 kt de la tormenta, velocidad de traslación y latitud.

**Evaluación honesta.** Se entrena con temporadas anteriores a `--anio-prueba` (2020 por defecto) y se evalúa con las posteriores, para que ninguna tormenta de prueba se filtre al entrenamiento. Se reportan ROC AUC, PR AUC, Brier y, a nivel operativo, cuántas alertas emite cada método, qué porcentaje acierta (precisión) y qué porcentaje de las ubicaciones afectadas detecta (cobertura).

## Uso

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m huracan_ia descargar                      # IBTrACS Atlántico y Pacífico (~300 MB)
python -m huracan_ia entrenar                       # construye el set, entrena y evalúa
python -m huracan_ia simular --tormenta OTIS --temporada 2023 --emision "2023-10-24 00:00" --boletin
python -m huracan_ia alertar --activas --boletin    # con las tormentas activas del NHC
streamlit run app/streamlit_app.py                  # visor de resultados
```

Con tu propia cartera: `--ubicaciones mis_ubicaciones.csv` (columnas `id, lat, lon` y opcionalmente `nombre, ciudad, estado`; también acepta el CSV del DENUE con `latitud, longitud`).

El boletín con Claude necesita `ANTHROPIC_API_KEY` (ver `.env.example`). Sin ella se genera un boletín de plantilla.

Salidas en `salidas/`: `modelo.joblib`, `metricas.json` y, por cada emisión, `*_alertas.csv`, `*_mapa.html` y `*_boletin.md`.

## Resultados

Corrida del 26 de octubre de 2026 con `python -m huracan_ia entrenar`: IBTrACS v04r01 (Atlántico Norte y Pacífico Oriental, 2004 en adelante), 535 tormentas que pasaron cerca de México y la cartera simulada de 400 ubicaciones. Entrenamiento con temporadas 2004–2019 (777,440 filas) y prueba con 2020 en adelante (379,535 filas de 153 tormentas, 9,092 positivos, 2.4 %). Las cifras salen de `salidas/metricas.json`.

Los umbrales del modelo se calibran sin ver la prueba: se entrena una versión con 2004–2015 y, con las temporadas 2016–2019, se elige la probabilidad con la que el nivel Medio detecta 80 % de las ubicaciones afectadas y el Alto 50 % (`--cobertura-medio`, `--cobertura-alto`). Salieron Medio ≥ 0.018 y Alto ≥ 0.228. Después se reentrena con 2004–2019 y se evalúa en 2020 en adelante.

| Método | ROC AUC | PR AUC | Alertas emitidas | Precisión | Cobertura |
| --- | --- | --- | --- | --- | --- |
| Índice de pesos fijos (nivel Medio o Alto) | 0.824 | 0.230 | 82,877 | 8.4 % | 76.8 % |
| Modelo LightGBM, nivel Medio o Alto (≥ 0.018) | 0.919 | 0.365 | 45,866 | 14.5 % | 73.4 % |
| Modelo LightGBM, solo nivel Alto (≥ 0.228) | 0.919 | 0.365 | 10,223 | 37.2 % | 41.8 % |
| Modelo LightGBM con la misma cobertura que el índice (≥ 0.014, referencia) | 0.919 | 0.365 | 51,923 | 13.4 % | 76.8 % |

Brier del modelo: 0.0195.

**Lectura.** El modelo ordena mejor las ubicaciones que el índice fijo (ROC AUC 0.92 contra 0.82, PR AUC 0.37 contra 0.23). Con los umbrales calibrados emite 45 % menos alertas que el índice y su precisión casi se duplica (14.5 % contra 8.4 %), a cambio de detectar 73 % de las ubicaciones afectadas en vez de 77 %. La cobertura en prueba quedó por debajo del 80 % buscado en validación: las temporadas recientes no se comportan exactamente igual que las de calibración. El nivel Alto es para priorizar: acierta más de un tercio de sus alertas.

Antes de calibrar, el umbral fijo de 0.30 solo detectaba 37 % de las afectadas.

Las variables con más peso (ganancia) son la distancia actual al centro, la velocidad de traslación, el margen respecto al cono y la distancia mínima al pronóstico.

**Backtest de Otis 2023** (`simular --tormenta OTIS --temporada 2023 --emision "2023-10-24 00:00"`, unas 30 horas antes de tocar tierra en Acapulco, cuando aún era tormenta tropical): de las 400 ubicaciones, 26 recibieron 34 kt o más en las 72 h siguientes. El índice fijo alertó 201 y atrapó las 26. El modelo con umbrales calibrados alertó 74 (73 en Medio, 1 en Alto) y también atrapó las 26: un tercio de las alertas del índice con la misma detección. Con el umbral anterior de 0.30 no alertaba ninguna. Solo una ubicación llegó a Alto porque Otis se intensificó de forma explosiva, algo que un pronóstico por persistencia no puede anticipar.

## Limitaciones

- **Umbrales calibrados con otro modelo.** Se eligen con el modelo entrenado hasta 2015 y se aplican al reentrenado hasta 2019, así que la cobertura real puede moverse unos puntos (en prueba fue 73 % contra 80 % buscado).
- **Persistencia vs. pronóstico oficial.** Para entrenar se usa un pronóstico por persistencia (extrapolar las últimas 12 h), porque es lo que se puede reconstruir para cualquier momento histórico. En operación se usa el pronóstico del NHC, que es mejor, así que el modelo probablemente sobrestima la incertidumbre. La siguiente fase es entrenar con el archivo de pronósticos oficiales del NHC.
- **Radio de 34 kt en operación.** El shapefile de puntos del NHC no trae radios de viento; si no se pasa `--r34-km`, se usa la mediana histórica.
- **Viento, no daño.** La etiqueta es haber recibido vientos de 34 kt, no daño ni cierre. Con registros reales de afectaciones se puede entrenar una versión con esa etiqueta.
- **Marea de tormenta** no está modelada. La lluvia se trata en el módulo `inundacion` (abajo).

## Zonas inundables por lluvia de huracanes (`inundacion/`)

Segundo módulo: **qué zonas se inundan, cuándo y con cuánta agua**, a partir de la lluvia que se espera de una tormenta. Piloto: Acapulco con el huracán John (septiembre de 2024), que dejó casi 1,000 mm en algunas estaciones y la peor inundación reciente del puerto.

```mermaid
flowchart LR
    A[IBTrACS + CMORPH<br/>lluvia satelital 2004-2026] --> B[Modelo de lluvia<br/>LightGBM esperado y P90]
    N[Aviso del NHC<br/>trayectoria pronosticada] --> B
    B --> C[Lluvia horaria<br/>sobre la ciudad]
    D[DEM Copernicus 30 m] --> E[Drenaje y HAND<br/>altura sobre el cauce]
    W[ESA WorldCover] --> F
    C --> F[Escurrimiento<br/>número de curva]
    E --> G[Curva de gasto<br/>por tramo]
    F --> H[Caudal por tramo<br/>hidrograma unitario]
    G --> I[Tirante hora por hora]
    H --> I
    I --> J[Animación web · NetCDF con tiempo<br/>GeoTIFF · inmuebles afectados]
```

**1. Modelo de lluvia (la parte de IA).** Responde: dada la posición, intensidad y movimiento de la tormenta, ¿cuántos mm/h van a caer en un punto? Se entrena con 2 millones de celdas-hora de lluvia satelital CMORPH alrededor de 367 tormentas que pasaron cerca de México (2004-2020) y se evalúa con 2021 en adelante. Las variables incluyen distancia, lado y frente de la tormenta, viento, intensificación, velocidad, relieve y cuánto llovía en la tormenta **cuando se emitió el pronóstico**, con el plazo como variable, así que funciona igual con un aviso real del NHC. La referencia es R-CLIPER, el modelo climatológico de lluvia que usa el NHC.

**2. Hidrología sobre el terreno.** Esquema HAND + curva de gasto sintética, como el de la NOAA (OWP): el escurrimiento por número de curva según la cobertura de suelo se convierte en caudal por tramo de cauce con un hidrograma unitario. Una curva de gasto construida con el propio terreno da el tirante, y se inunda toda celda cuya altura sobre el cauce quede por debajo.

### Resultados

Lluvia acumulada por evento en celdas de tierra a menos de 300 km de la trayectoria (70 tormentas de prueba, 2021+, pronóstico emitido 24 h antes):

| | Correlación | RMSE (mm) | Sesgo en celdas con ≥100 mm |
|---|---|---|---|
| R-CLIPER (NHC) | 0.54 | 54.5 | −102 mm |
| **LightGBM esperado** | **0.64** | **53.1** | −108 mm |
| **LightGBM severo (P90)** | 0.67 | 76.5 | **+17 mm** |

El modelo esperado explica mejor dónde llueve, pero, como R-CLIPER, se queda corto en la lluvia extrema. El escenario severo (cuantil 90, cobertura real de 89% por hora) corrige casi todo ese sesgo y es el que conviene usar para alertar.

John 2024 en Acapulco (`python -m inundacion simular ...`, 250 inmuebles **simulados**):

| Escenario | Lluvia total | Área inundada máx. | Inmuebles con ≥0.15 m |
|---|---|---|---|
| Lluvia observada (CMORPH) | 686 mm | 43.8 km² | 15 |
| Aviso NHC 23 sep 12:00 (esperado / severo) | 54 / 177 mm | 7.6 / 17.3 km² | 0 / 5 |
| Aviso NHC 25 sep 09:00 (esperado / severo) | 170 / 365 mm | 18.1 / 26.8 km² | 5 / 7 |
| Trayectoria real + modelo | 207 mm | 28.0 km² | 7 |

Con la lluvia observada, las zonas inundadas son el río La Sabana y la planicie de Llano Largo y Diamante, con el pico el 26 de septiembre, lo que coincide con lo reportado. Con los avisos que había en su momento, ningún escenario anticipa la magnitud: el NHC pronosticaba la trayectoria de John lejos de Acapulco y no previó su regeneración frente a Guerrero, y la lluvia en el puerto cayó con el centro a 200-300 km. Para este evento, el error de trayectoria pesó más que el modelo de lluvia.

### Uso

```bash
python -m inundacion terreno --dominio acapulco          # DEM, cobertura, drenaje y HAND
python -m inundacion entrenar-lluvia                     # ~6,000 horas de CMORPH (~1.2 GB), LightGBM vs R-CLIPER
python -m inundacion simular --tormenta JOHN --temporada 2024 \
    --inicio "2024-09-22 18:00" --fin "2024-09-29 12:00" \
    --nhc datos/inundacion/nhc/ep102024_5day_005.zip --nhc datos/inundacion/nhc/ep102024_5day_011.zip \
    --emision-real "2024-09-23 18:00"
```

Los avisos se descargan de `https://www.nhc.noaa.gov/gis/forecast/archive/` (`ep102024_5day_NNN.zip`). Salidas en `salidas/inundacion/<tormenta>_<dominio>/`:

- `animacion.html`: mapa animado con barra de tiempo, hietograma, área inundada y los inmuebles cambiando de color hora por hora.
- `escenarioN_tirante_horario.nc`: tirante por hora (NetCDF CF con tiempo). Se abre en ArcGIS Pro o QGIS como ráster multidimensional, listo para el Time Slider.
- `escenarioN_tirante_max.tif` y `escenarioN_inmuebles.gpkg` / `.csv`, con el tirante máximo y la hora en que se inunda cada inmueble.
- `arcgis/InundacionHuracan.pyt`: caja de herramientas de ArcGIS Pro que carga un escenario con su simbología.

### Limitaciones

- **Modelo de primer orden.** HAND + curva de gasto no simula flujo 2D, remanso ni drenaje urbano. Tampoco incluye la marea de tormenta, que en Acapulco se suma en la costa.
- **DEM de superficie.** Copernicus incluye edificios y árboles; en zona urbana un DEM de terreno desnudo (LiDAR de INEGI, FABDEM) daría mejores tirantes.
- **Lluvia uniforme** sobre el dominio, tomada en su centro (las celdas de CMORPH son de ~27 km).
- **Sin validación contra manchas observadas.** No hubo acceso a polígonos de inundación (CENAPRED, Sentinel-1). Con ellos se puede calibrar el número de curva y la rugosidad, y entrenar un clasificador de celdas inundadas.
- **Lluvia extrema.** El modelo esperado subestima eventos como John. Para alertar conviene usar el escenario P90.

## Estructura

```
huracan_ia/
  geo.py          distancias y rumbos geodésicos (sin arcpy)
  ibtracs.py      descarga, lectura e interpolación horaria de IBTrACS
  etiquetas.py    viento recibido por ubicación según radios por cuadrante
  pronostico.py   pronóstico por persistencia y lectura del NHC
  variables.py    variables por ubicación
  indice_base.py  índice de pesos fijos (línea base)
  dataset.py      conjunto de entrenamiento histórico
  modelo.py       entrenamiento, evaluación y niveles
  boletin.py      boletín ejecutivo con Claude
  mapa.py         mapa Folium
  ubicaciones.py  cartera simulada o desde CSV/DENUE
inundacion/
  terreno.py      DEM Copernicus, cobertura WorldCover, drenaje y HAND (pysheds)
  lluvia.py       CMORPH horario y elevación gruesa
  modelo_lluvia.py  LightGBM (esperado y P90) vs R-CLIPER, evaluación por evento
  escenarios.py   lluvia observada, por trayectoria y por aviso del NHC
  hidrologia.py   escurrimiento, caudal, curva de gasto y tirante por hora
  exposicion.py   inmuebles simulados y su tirante hora por hora
  exportar.py     NetCDF con tiempo, GeoTIFF y GeoPackage
  animacion.py    mapa animado (Leaflet)
arcgis/InundacionHuracan.pyt  carga de escenarios en ArcGIS Pro
app/streamlit_app.py
tests/            pruebas con tormentas y terrenos sintéticos (pytest)
```

## Fuentes

- [IBTrACS v04r01](https://www.ncei.noaa.gov/products/international-best-track-archive), NOAA/NCEI.
- [NHC GIS](https://www.nhc.noaa.gov/gis/) y [CurrentStorms.json](https://www.nhc.noaa.gov/CurrentStorms.json), NOAA.
- [Cono de pronóstico del NHC](https://www.nhc.noaa.gov/aboutcone.shtml).
- [DENUE](https://www.inegi.org.mx/app/mapa/denue/), INEGI (opcional).
- [Copernicus DEM GLO-30](https://registry.opendata.aws/copernicus-dem/), ESA, en AWS Open Data.
- [ESA WorldCover 2021](https://esa-worldcover.org/), 10 m.
- [CMORPH CDR](https://www.ncei.noaa.gov/products/climate-data-records/precipitation-cmorph), NOAA/NCEI.
- R-CLIPER: Tuleya et al. (2007), *Weather and Forecasting* 22, 56-70. HAND: Nobre et al. (2011); curva de gasto sintética: Zheng et al. (2018), NOAA OWP.
