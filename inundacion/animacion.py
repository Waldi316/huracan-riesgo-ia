"""Animación web de la inundación: un HTML autocontenido (Leaflet) con barra de tiempo.

Cada hora es una imagen PNG transparente del tirante, reproyectada a lat/lon y
superpuesta al mapa base. Los inmuebles cambian de color según el agua que tienen
en esa hora. Se pueden incluir varios escenarios (lluvia observada, pronóstico...)
y alternar entre ellos.
"""
import base64
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, exposicion, hidrologia

TIRANTE_MAX_ESCALA_M = 3.0
ANCHO_IMAGEN_PX = 640
ZONA_HORARIA = "America/Mexico_City"


def _malla_latlon(ter, ancho_px=ANCHO_IMAGEN_PX):
    """Índices planos del ráster UTM para cada píxel de una malla regular lat/lon que cubre el dominio."""
    from rasterio.transform import Affine, array_bounds
    from rasterio.warp import transform_bounds

    alto, ancho = ter["dem"].shape
    oeste, sur, este, norte = transform_bounds(ter["crs"], "EPSG:4326", *array_bounds(alto, ancho, Affine(*ter["transform"])))
    alto_px = int(round(ancho_px * (norte - sur) / ((este - oeste) * np.cos(np.radians((norte + sur) / 2)))))
    lons = oeste + (np.arange(ancho_px) + 0.5) * (este - oeste) / ancho_px
    lats = norte - (np.arange(alto_px) + 0.5) * (norte - sur) / alto_px
    lat_m, lon_m = np.meshgrid(lats, lons, indexing="ij")
    filas, cols = exposicion.a_fila_columna(ter, lat_m.ravel(), lon_m.ravel())
    plano = (filas * ancho + cols).reshape(alto_px, ancho_px)
    return plano, [[sur, oeste], [norte, este]]


def _colores(tirante):
    """Rampa de azules (claro = poco, oscuro = mucho), transparente donde no hay agua."""
    paradas = np.array([0.15, 0.5, 1.0, 2.0, TIRANTE_MAX_ESCALA_M])
    rgb = np.array([[198, 219, 239], [107, 174, 214], [33, 113, 181], [8, 69, 148], [8, 29, 88]], dtype=float)
    t = np.nan_to_num(tirante, nan=0.0)
    img = np.zeros(t.shape + (4,), dtype=np.uint8)
    for c in range(3):
        img[..., c] = np.interp(t, paradas, rgb[:, c]).astype(np.uint8)
    img[..., 3] = np.where(np.isfinite(tirante), 215, 0)
    return img


def _png_base64(img):
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(img, "RGBA").save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _cuadros(ter, sim, plano, horas):
    return [_png_base64(_colores(hidrologia.tirante_en_celdas(ter, sim, k).ravel()[plano])) for k in horas]


def generar(ter, escenarios, ruta_html, titulo, horas=None, trayectorias=None, nota=""):
    """escenarios: lista de dicts {nombre, sim, res, tirante_inmuebles} con la misma serie de horas.

    trayectorias: lista opcional de {nombre, lat, lon} para dibujar la tormenta.
    """
    plano, limites = _malla_latlon(ter)
    sim0 = escenarios[0]["sim"]
    if horas is None:
        horas = sorted(set().union(*[set(hidrologia_horas_activas(e["sim"])) for e in escenarios]))
    tiempos = pd.DatetimeIndex(sim0["horas"])[list(horas)]
    locales = tiempos.tz_localize("UTC").tz_convert(ZONA_HORARIA)
    cartera = escenarios[0]["res"]
    datos = {
        "titulo": titulo,
        "nota": nota,
        "limites": limites,
        "horas": [f"{t:%d %b %H:%M}" for t in locales],
        "inmuebles": {"id": cartera["id"].tolist(), "lat": cartera["lat"].round(5).tolist(),
                      "lon": cartera["lon"].round(5).tolist()},
        "escenarios": [],
        "trayectorias": trayectorias or [],
        "umbral_alto": exposicion.NIVELES[0][0],
    }
    for e in escenarios:
        area = exposicion.area_inundada_km2(ter, e["sim"])
        datos["escenarios"].append({
            "nombre": e["nombre"],
            "cuadros": _cuadros(ter, e["sim"], plano, horas),
            "lluvia": np.round(e["sim"]["lluvia_mm_h"][list(horas)], 1).tolist(),
            "acumulada": np.round(np.cumsum(e["sim"]["lluvia_mm_h"])[list(horas)], 0).tolist(),
            "area": np.round(area[list(horas)], 2).tolist(),
            "tirante_cm": np.round(e["tirante_inmuebles"][:, list(horas)] * 100).astype(int).tolist(),
        })
    html = PLANTILLA.replace("__DATOS__", json.dumps(datos, ensure_ascii=False, separators=(",", ":")))
    Path(ruta_html).write_text(html, encoding="utf-8")
    return ruta_html


def hidrologia_horas_activas(sim, margen=6):
    from .exportar import horas_con_agua

    return horas_con_agua(sim, margen)


PLANTILLA = r"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Inundación simulada</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<style>
  :root { --fondo:#ffffff; --texto:#1b1f24; --tenue:#5b6470; --borde:#d9dde3; --acento:#2171b5;
          --alto:#c0392b; --medio:#e67e22; --nada:#8a939e; --panel:rgba(255,255,255,.95); }
  @media (prefers-color-scheme: dark) { :root { --fondo:#15181c; --texto:#e8eaed; --tenue:#9aa3ad; --borde:#2c3138;
          --acento:#6baed6; --panel:rgba(24,27,31,.95); } }
  * { box-sizing:border-box; }
  html, body { margin:0; height:100%; background:var(--fondo); color:var(--texto);
               font:14px/1.4 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
  #mapa { position:absolute; inset:0; }
  .panel { position:absolute; z-index:1000; background:var(--panel); border:1px solid var(--borde);
           border-radius:10px; padding:12px 14px; box-shadow:0 2px 10px rgba(0,0,0,.12); }
  #cabecera { top:12px; left:56px; right:12px; max-width:440px; }
  #cabecera h1 { font-size:16px; margin:0 0 4px; }
  #cabecera p { margin:0; color:var(--tenue); font-size:12px; }
  #escenarios { display:flex; gap:6px; flex-wrap:wrap; margin-top:8px; }
  #escenarios button { border:1px solid var(--borde); background:transparent; color:var(--texto);
                       border-radius:999px; padding:4px 10px; font-size:12px; cursor:pointer; }
  #escenarios button.activo { background:var(--acento); border-color:var(--acento); color:#fff; }
  #control { bottom:16px; left:12px; right:12px; max-width:640px; margin:0 auto; }
  #fila { display:flex; align-items:center; gap:10px; }
  #play { width:36px; height:36px; border-radius:50%; border:none; background:var(--acento); color:#fff;
          font-size:15px; cursor:pointer; flex:none; }
  #barra { flex:1; accent-color:var(--acento); }
  #hora { font-variant-numeric:tabular-nums; font-weight:600; min-width:96px; text-align:right; }
  #lluvia { width:100%; height:46px; margin-top:8px; display:block; }
  #cifras { display:flex; gap:16px; flex-wrap:wrap; margin-top:6px; font-size:12px; color:var(--tenue); }
  #cifras b { color:var(--texto); font-variant-numeric:tabular-nums; }
  #leyenda { top:12px; right:12px; font-size:12px; display:none; }
  @media (min-width:760px) { #leyenda { display:block; } #cabecera { right:auto; } }
  .rampa { width:140px; height:10px; border-radius:3px;
           background:linear-gradient(90deg,#c6dbef,#6baed6,#2171b5,#08458f,#081d58); }
  .esc { display:flex; justify-content:space-between; color:var(--tenue); }
  .punto { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:4px; vertical-align:-1px; }
</style>
</head>
<body>
<div id="mapa"></div>
<div id="cabecera" class="panel"><h1 id="titulo"></h1><p id="nota"></p><div id="escenarios"></div></div>
<div id="leyenda" class="panel">
  <div style="font-weight:600;margin-bottom:4px">Tirante (m)</div>
  <div class="rampa"></div><div class="esc"><span>0.15</span><span>1</span><span>3+</span></div>
  <div style="margin-top:8px"><span class="punto" style="background:var(--alto)"></span>≥ 0.5 m
  <span class="punto" style="background:var(--medio);margin-left:8px"></span>0.15–0.5 m</div>
  <div><span class="punto" style="background:var(--nada)"></span>Inmueble seco</div>
</div>
<div id="control" class="panel">
  <div id="fila"><button id="play" aria-label="Reproducir">▶</button>
    <input id="barra" type="range" min="0" value="0" step="1"><span id="hora"></span></div>
  <svg id="lluvia" preserveAspectRatio="none"></svg>
  <div id="cifras"><span>Lluvia en la hora: <b id="c-lluvia"></b></span><span>Acumulada: <b id="c-acum"></b></span>
    <span>Área inundada: <b id="c-area"></b></span><span>Inmuebles con agua: <b id="c-inm"></b></span></div>
</div>
<script>
const D = __DATOS__;
const mapa = L.map('mapa', {zoomControl:true}).fitBounds(D.limites);
const claro = L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
  {attribution:'&copy; OpenStreetMap &copy; CARTO', maxZoom:18});
const sat = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
  {attribution:'Esri World Imagery', maxZoom:18}).addTo(mapa);
L.control.layers({'Satélite':sat, 'Mapa claro':claro}, null, {position:'bottomright'}).addTo(mapa);
D.trayectorias.forEach((t,i) => L.polyline(t.lat.map((la,k)=>[la,t.lon[k]]),
  {color:i ? '#f1c40f' : '#ffffff', weight:2, dashArray:i ? '6 6' : null}).bindTooltip(t.nombre).addTo(mapa));
const capa = L.imageOverlay('data:image/png;base64,' + D.escenarios[0].cuadros[0], D.limites, {opacity:1}).addTo(mapa);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const puntos = D.inmuebles.lat.map((la,i) => L.circleMarker([la, D.inmuebles.lon[i]],
  {radius:4, weight:1, color:'#fff', fillOpacity:.95}).addTo(mapa));
document.getElementById('titulo').textContent = D.titulo;
document.getElementById('nota').textContent = D.nota;
let esc = 0, k = 0, timer = null;
const barra = document.getElementById('barra'); barra.max = D.horas.length - 1;
const caja = document.getElementById('escenarios');
D.escenarios.forEach((e,i) => { const b = document.createElement('button'); b.textContent = e.nombre;
  b.onclick = () => { esc = i; [...caja.children].forEach((x,j)=>x.classList.toggle('activo', j===i)); dibujarLluvia(); pintar(); };
  caja.appendChild(b); });
caja.children[0].classList.add('activo');
const svg = document.getElementById('lluvia');
function dibujarLluvia() {
  const e = D.escenarios[esc], n = e.lluvia.length, mx = Math.max(5, ...D.escenarios.map(x=>Math.max(...x.lluvia)));
  svg.setAttribute('viewBox', `0 0 ${n} 100`);
  svg.innerHTML = e.lluvia.map((v,i)=>`<rect x="${i}" y="${100-v/mx*100}" width="0.9" height="${v/mx*100}" fill="${css('--acento')}" opacity=".55"/>`).join('')
    + `<line id="cursor" x1="0" x2="0" y1="0" y2="100" stroke="${css('--texto')}" stroke-width="0.6" vector-effect="non-scaling-stroke"/>`;
}
function pintar() {
  const e = D.escenarios[esc];
  capa.setUrl('data:image/png;base64,' + e.cuadros[k]);
  barra.value = k; document.getElementById('hora').textContent = D.horas[k];
  const c = document.getElementById('cursor'); c.setAttribute('x1', k + .45); c.setAttribute('x2', k + .45);
  document.getElementById('c-lluvia').textContent = e.lluvia[k].toFixed(1) + ' mm';
  document.getElementById('c-acum').textContent = e.acumulada[k] + ' mm';
  document.getElementById('c-area').textContent = e.area[k].toFixed(1) + ' km²';
  let n = 0;
  puntos.forEach((p,i) => { const cm = e.tirante_cm[i][k];
    const col = cm >= D.umbral_alto*100 ? css('--alto') : cm > 0 ? css('--medio') : css('--nada');
    if (cm > 0) n++;
    p.setStyle({fillColor:col, radius: cm > 0 ? 5 : 3.5});
    p.bindTooltip(`${D.inmuebles.id[i]} · ${(cm/100).toFixed(2)} m`); });
  document.getElementById('c-inm').textContent = `${n} de ${puntos.length}`;
}
barra.oninput = () => { k = +barra.value; pintar(); };
document.getElementById('play').onclick = () => {
  const b = document.getElementById('play');
  if (timer) { clearInterval(timer); timer = null; b.textContent = '▶'; return; }
  b.textContent = '❚❚';
  timer = setInterval(() => { k = (k + 1) % D.horas.length; pintar(); }, 180);
};
dibujarLluvia(); pintar();
</script>
</body>
</html>
"""
