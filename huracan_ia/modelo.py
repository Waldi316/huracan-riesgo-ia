"""Entrenamiento, evaluación y uso del modelo de probabilidad de viento."""
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, precision_score, recall_score, roc_auc_score

from . import config
from .variables import COLUMNAS_MODELO

OBJETIVO = "y34"


def separar_por_temporada(datos, anio_prueba):
    """Entrena con temporadas anteriores y evalúa con las más recientes (sin fuga entre tormentas)."""
    return datos[datos["temporada"] < anio_prueba], datos[datos["temporada"] >= anio_prueba]


def entrenar(entrenamiento, objetivo=OBJETIVO, semilla=7):
    modelo = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.9,
        random_state=semilla,
        verbose=-1,
    )
    modelo.fit(entrenamiento[COLUMNAS_MODELO], entrenamiento[objetivo])
    return modelo


def predecir(modelo, variables):
    return modelo.predict_proba(variables[COLUMNAS_MODELO])[:, 1]


def evaluar(modelo, prueba, objetivo=OBJETIVO):
    """Compara el modelo contra el índice de pesos fijos en el mismo conjunto de prueba."""
    y = prueba[objetivo].to_numpy()
    prob = predecir(modelo, prueba)
    ifa = prueba["ifa"].to_numpy()

    # Alerta operativa: el índice base alerta en Medio o Alto; el modelo, con prob >= umbral medio.
    alerta_base = prueba["nivel_base"].isin(["Medio", "Alto"]).to_numpy()
    alerta_modelo = prob >= config.UMBRAL_PROB_MEDIO

    def resumen_alerta(alerta):
        return {
            "alertas_emitidas": int(alerta.sum()),
            "precision": float(precision_score(y, alerta, zero_division=0)),
            "cobertura": float(recall_score(y, alerta, zero_division=0)),
        }

    # Comparación justa: el umbral del modelo que alcanza la misma cobertura que el índice base.
    cobertura_base = recall_score(y, alerta_base, zero_division=0)
    umbral_igual = float(np.quantile(prob[y == 1], 1 - cobertura_base)) if y.sum() else 1.0
    alerta_igual = prob >= umbral_igual

    return {
        "filas_prueba": int(len(y)),
        "positivos_prueba": int(y.sum()),
        "tormentas_prueba": int(prueba["sid"].nunique()),
        "modelo": {
            "roc_auc": float(roc_auc_score(y, prob)),
            "pr_auc": float(average_precision_score(y, prob)),
            "brier": float(brier_score_loss(y, prob)),
            **resumen_alerta(alerta_modelo),
        },
        "modelo_misma_cobertura_que_base": {
            "umbral_prob": umbral_igual,
            **resumen_alerta(alerta_igual),
        },
        "indice_base": {
            "roc_auc": float(roc_auc_score(y, ifa)),
            "pr_auc": float(average_precision_score(y, ifa)),
            **resumen_alerta(alerta_base),
        },
    }


def importancia(modelo):
    return (
        pd.Series(modelo.booster_.feature_importance("gain"), index=COLUMNAS_MODELO)
        .sort_values(ascending=False)
        .round(1)
        .to_dict()
    )


def guardar(modelo, metricas, entrenamiento, carpeta=config.CARPETA_SALIDAS):
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    meta = {
        "objetivo": OBJETIVO,
        "columnas": COLUMNAS_MODELO,
        # Para pronósticos del NHC sin radio de 34 kt se usa la mediana histórica.
        "r34_mediana_km": float(np.nanmedian(entrenamiento["r34_max_km"])),
        "temporadas_entrenamiento": sorted(int(t) for t in entrenamiento["temporada"].unique()),
    }
    joblib.dump({"modelo": modelo, "meta": meta}, carpeta / "modelo.joblib")
    (carpeta / "metricas.json").write_text(json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8")


def cargar(ruta=config.CARPETA_SALIDAS / "modelo.joblib"):
    paquete = joblib.load(ruta)
    return paquete["modelo"], paquete["meta"]


def nivel(prob):
    prob = np.asarray(prob)
    return np.where(prob >= config.UMBRAL_PROB_ALTO, "Alto", np.where(prob >= config.UMBRAL_PROB_MEDIO, "Medio", "Bajo"))
