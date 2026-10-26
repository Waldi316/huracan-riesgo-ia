"""Mapa HTML (Folium) con la trayectoria pronosticada y las ubicaciones por nivel."""
import folium

COLORES = {"Alto": "#b2182b", "Medio": "#ef8a62", "Bajo": "#4393c3"}


def generar(resultados, pronostico, nombre_tormenta, ruta_html, trayectoria_real=None):
    centro = [pronostico["lat"].iloc[0], pronostico["lon"].iloc[0]]
    m = folium.Map(location=centro, zoom_start=5, tiles="CartoDB positron")

    folium.PolyLine(list(zip(pronostico["lat"], pronostico["lon"])), color="#333", weight=3,
                    dash_array="6", tooltip="Pronóstico").add_to(m)
    for fila in pronostico.itertuples():
        folium.CircleMarker([fila.lat, fila.lon], radius=3, color="#333",
                            tooltip=f"+{int(fila.horas)} h · {int(fila.vmax_kt)} kt").add_to(m)
    if trayectoria_real is not None:
        folium.PolyLine(list(zip(trayectoria_real["lat"], trayectoria_real["lon"])), color="#7b3294",
                        weight=2, tooltip="Trayectoria real (IBTrACS)").add_to(m)

    for nivel in ("Bajo", "Medio", "Alto"):
        capa = folium.FeatureGroup(name=f"Nivel {nivel}")
        for r in resultados[resultados["nivel"] == nivel].itertuples():
            folium.CircleMarker(
                [r.lat_ubicacion, r.lon_ubicacion], radius=5, color=COLORES[nivel], fill=True, fill_opacity=0.85,
                tooltip=f"{r.id} · {r.ciudad} · prob {r.prob:.0%} · índice base {r.nivel_base}",
            ).add_to(capa)
        capa.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(
        f"<div style='position:fixed;top:10px;left:50px;z-index:999;background:white;padding:6px 10px;"
        f"border-radius:4px;font-family:sans-serif'><b>{nombre_tormenta}</b></div>"
    ))
    m.save(str(ruta_html))
    return ruta_html
