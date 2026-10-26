"""Boletín ejecutivo redactado por Claude a partir del resultado del modelo.

El LLM no calcula el riesgo: recibe un resumen estructurado (conteos, top de
ubicaciones, horas al arribo) y lo convierte en un boletín claro con acciones.
Si no hay credenciales de la API, se genera un boletín de plantilla.
"""
import json

import anthropic

from . import config

INSTRUCCIONES = """Eres analista de inteligencia geográfica de una cadena de inmuebles en México.
Redacta un boletín ejecutivo en español, en Markdown, a partir del JSON que recibes.

Estructura:
1. Título con el nombre de la tormenta y la hora de emisión.
2. Situación en dos o tres frases: intensidad, movimiento y cuándo llega a tierra o pasa más cerca.
3. Tabla con las ubicaciones de nivel Alto (id, ciudad, estado, probabilidad, horas al punto más cercano).
4. Acciones recomendadas por nivel (Alto, Medio), concretas y con plazos en horas.
5. Una línea final que aclare que la probabilidad es la de recibir vientos de tormenta tropical (34 kt o más) en las próximas 72 horas.

Usa solo los datos del JSON. No inventes ubicaciones, cifras ni pronósticos. Si no hay ubicaciones en nivel Alto, dilo."""


def resumen_para_boletin(nombre_tormenta, emision, estado, resultados):
    """Arma el JSON que recibe el LLM. resultados: DataFrame con prob, nivel, horas_a_minima, ciudad, estado."""
    ordenados = resultados.sort_values("prob", ascending=False)
    top = ordenados[ordenados["nivel"].isin(["Alto", "Medio"])].head(config.MAXIMO_UBICACIONES_EN_BOLETIN)
    por_estado = (
        ordenados[ordenados["nivel"] != "Bajo"].groupby(["estado", "nivel"]).size().unstack(fill_value=0).to_dict("index")
    )
    return {
        "tormenta": nombre_tormenta,
        "emision": str(emision),
        "estado_actual": {
            "lat": round(estado["lat"], 2),
            "lon": round(estado["lon"], 2),
            "viento_max_kt": round(estado["vmax_kt"]),
            "velocidad_traslacion_kmh": round(estado["velocidad_kmh"], 1),
            "rumbo_grados": round(estado["rumbo"]),
        },
        "conteo_por_nivel": ordenados["nivel"].value_counts().to_dict(),
        "conteo_por_estado": por_estado,
        "ubicaciones_prioritarias": [
            {
                "id": r.id,
                "ciudad": r.ciudad,
                "estado": r.estado,
                "nivel": r.nivel,
                "probabilidad": round(float(r.prob), 2),
                "horas_al_punto_mas_cercano": round(float(r.horas_a_minima)),
                "distancia_minima_km": round(float(r.dist_min_km)),
            }
            for r in top.itertuples()
        ],
    }


def generar(resumen, modelo=config.MODELO_CLAUDE):
    """Devuelve el boletín en Markdown. Usa Claude si hay credenciales; si no, una plantilla."""
    try:
        cliente = anthropic.Anthropic()
        respuesta = cliente.beta.messages.create(
            model=modelo,
            max_tokens=4000,
            system=INSTRUCCIONES,
            output_config={"effort": "low"},
            # Si el modelo principal declina la solicitud, la API la reintenta con otro modelo.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": json.dumps(resumen, ensure_ascii=False, indent=2)}],
        )
    except (anthropic.AuthenticationError, TypeError):
        # TypeError: el SDK no encontró ninguna credencial (ANTHROPIC_API_KEY, token o perfil).
        print("[AVISO] Sin credenciales válidas de la API de Claude; se usa el boletín de plantilla.")
        return plantilla(resumen)
    except anthropic.APIConnectionError:
        print("[AVISO] No se pudo conectar con la API de Claude; se usa el boletín de plantilla.")
        return plantilla(resumen)

    if respuesta.stop_reason == "refusal":
        print("[AVISO] La API declinó la solicitud; se usa el boletín de plantilla.")
        return plantilla(resumen)
    return "".join(b.text for b in respuesta.content if b.type == "text")


def plantilla(resumen):
    e = resumen["estado_actual"]
    lineas = [
        f"# Boletín: {resumen['tormenta']} ({resumen['emision']})",
        "",
        f"Posición {e['lat']}, {e['lon']}; viento máximo {e['viento_max_kt']} kt; "
        f"avanza a {e['velocidad_traslacion_kmh']} km/h con rumbo {e['rumbo_grados']}°.",
        "",
        "| Nivel | Ubicaciones |",
        "| --- | --- |",
    ]
    for nivel in ("Alto", "Medio", "Bajo"):
        lineas.append(f"| {nivel} | {resumen['conteo_por_nivel'].get(nivel, 0)} |")
    if resumen["ubicaciones_prioritarias"]:
        lineas += ["", "| Id | Ciudad | Estado | Nivel | Probabilidad | Horas |", "| --- | --- | --- | --- | --- | --- |"]
        for u in resumen["ubicaciones_prioritarias"]:
            lineas.append(
                f"| {u['id']} | {u['ciudad']} | {u['estado']} | {u['nivel']} | {u['probabilidad']:.0%} | {u['horas_al_punto_mas_cercano']} |"
            )
    lineas += ["", "_Probabilidad de recibir vientos de 34 kt o más en las próximas 72 horas._"]
    return "\n".join(lineas)
