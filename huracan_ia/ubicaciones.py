"""Cartera de ubicaciones a evaluar.

El proyecto no usa datos internos de ninguna empresa. Hay dos opciones:

1. Un CSV propio o del DENUE (INEGI) con columnas id, nombre, lat, lon.
2. Una cartera SIMULADA alrededor de ciudades mexicanas, para demos y pruebas.
"""
import numpy as np
import pandas as pd

# (ciudad, estado, lat, lon, peso relativo de ubicaciones)
CIUDADES = [
    ("Cancún", "Quintana Roo", 21.16, -86.85, 4), ("Playa del Carmen", "Quintana Roo", 20.63, -87.07, 2),
    ("Chetumal", "Quintana Roo", 18.50, -88.30, 2), ("Mérida", "Yucatán", 20.97, -89.62, 4),
    ("Campeche", "Campeche", 19.85, -90.53, 2), ("Ciudad del Carmen", "Campeche", 18.65, -91.83, 2),
    ("Villahermosa", "Tabasco", 17.99, -92.93, 3), ("Coatzacoalcos", "Veracruz", 18.14, -94.46, 2),
    ("Veracruz", "Veracruz", 19.17, -96.13, 4), ("Xalapa", "Veracruz", 19.54, -96.91, 2),
    ("Poza Rica", "Veracruz", 20.53, -97.46, 2), ("Tuxpan", "Veracruz", 20.95, -97.40, 1),
    ("Tampico", "Tamaulipas", 22.25, -97.86, 3), ("Ciudad Victoria", "Tamaulipas", 23.74, -99.15, 2),
    ("Matamoros", "Tamaulipas", 25.87, -97.50, 2), ("Reynosa", "Tamaulipas", 26.08, -98.29, 3),
    ("Monterrey", "Nuevo León", 25.67, -100.31, 6), ("San Luis Potosí", "San Luis Potosí", 22.15, -100.98, 3),
    ("Tapachula", "Chiapas", 14.90, -92.26, 2), ("Tuxtla Gutiérrez", "Chiapas", 16.75, -93.12, 3),
    ("Salina Cruz", "Oaxaca", 16.17, -95.20, 1), ("Oaxaca", "Oaxaca", 17.07, -96.72, 2),
    ("Puerto Escondido", "Oaxaca", 15.86, -97.07, 1), ("Acapulco", "Guerrero", 16.85, -99.82, 4),
    ("Chilpancingo", "Guerrero", 17.55, -99.50, 2), ("Zihuatanejo", "Guerrero", 17.64, -101.55, 1),
    ("Lázaro Cárdenas", "Michoacán", 17.96, -102.20, 1), ("Colima", "Colima", 19.24, -103.72, 1),
    ("Manzanillo", "Colima", 19.05, -104.32, 1), ("Guadalajara", "Jalisco", 20.67, -103.35, 6),
    ("Puerto Vallarta", "Jalisco", 20.65, -105.23, 2), ("Tepic", "Nayarit", 21.50, -104.89, 2),
    ("Mazatlán", "Sinaloa", 23.25, -106.41, 2), ("Culiacán", "Sinaloa", 24.80, -107.39, 3),
    ("Los Mochis", "Sinaloa", 25.79, -108.99, 2), ("La Paz", "Baja California Sur", 24.14, -110.31, 2),
    ("Los Cabos", "Baja California Sur", 22.89, -109.91, 2), ("Guaymas", "Sonora", 27.92, -110.90, 1),
    ("Hermosillo", "Sonora", 29.07, -110.96, 3), ("Ciudad de México", "Ciudad de México", 19.43, -99.13, 8),
]


def simuladas(n=400, semilla=7, dispersion_km=8.0):
    """Genera n ubicaciones ficticias repartidas según el peso de cada ciudad."""
    rng = np.random.default_rng(semilla)
    pesos = np.array([c[4] for c in CIUDADES], dtype=float)
    elegidas = rng.choice(len(CIUDADES), size=n, p=pesos / pesos.sum())
    sigma_grados = dispersion_km / 111.0
    filas = []
    for i, k in enumerate(elegidas):
        ciudad, estado, lat, lon, _ = CIUDADES[k]
        filas.append({
            "id": f"SIM-{i + 1:04d}",
            "nombre": f"Ubicación simulada {i + 1} ({ciudad})",
            "ciudad": ciudad,
            "estado": estado,
            "lat": lat + rng.normal(0, sigma_grados),
            "lon": lon + rng.normal(0, sigma_grados),
        })
    return pd.DataFrame(filas)


def desde_csv(ruta):
    """Lee un CSV con al menos id, lat, lon (acepta latitud/longitud del DENUE)."""
    df = pd.read_csv(ruta, encoding_errors="replace")
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={"latitud": "lat", "longitud": "lon", "nom_estab": "nombre", "entidad": "estado"})
    if "id" not in df.columns:
        df["id"] = [f"U-{i + 1:05d}" for i in range(len(df))]
    for col in ("nombre", "estado", "ciudad"):
        if col not in df.columns:
            df[col] = ""
    df = df.dropna(subset=["lat", "lon"])
    return df[["id", "nombre", "ciudad", "estado", "lat", "lon"]].reset_index(drop=True)
