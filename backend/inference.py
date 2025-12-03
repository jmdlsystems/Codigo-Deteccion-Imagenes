"""Funciones de inferencia para el modelo EKNN guardado con joblib.

Este módulo carga ``backend/modelos/modelo_final_eknn.pkl`` usando ``joblib`` y
preprocesa imágenes con OpenCV replicando el flujo de entrenamiento:

1. Carga la imagen (desde ruta o archivo subido).
2. Convierte a RGB y redimensiona a 224x224.
3. Normaliza a [0,1].
4. Extrae características finas y gruesas (EKNN) y las concatena en un vector.
5. Ejecuta la predicción con el modelo EKNN y devuelve la clase, vecinos y
   distancias.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from joblib import load

BASE_MODEL_PATH = Path(__file__).parent / "modelos" / "modelo_final_eknn.pkl"


def cargar_modelo(model_path: str | Path = BASE_MODEL_PATH):
    """Carga el modelo EKNN guardado con joblib."""

    ruta = Path(model_path).expanduser().resolve()
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el modelo en {ruta}")
    return load(ruta)


def _leer_imagen(ruta_o_archivo: str | Path | Any) -> np.ndarray:
    """Lee una imagen desde ruta o archivo (por ejemplo, Flask FileStorage)."""

    if hasattr(ruta_o_archivo, "read") or hasattr(ruta_o_archivo, "stream"):
        stream = getattr(ruta_o_archivo, "stream", ruta_o_archivo)
        if hasattr(stream, "seek"):
            stream.seek(0)
        data = stream.read()
        if hasattr(stream, "seek"):
            stream.seek(0)
        array = np.frombuffer(data, dtype=np.uint8)
        imagen = cv2.imdecode(array, cv2.IMREAD_COLOR)
    else:
        ruta = Path(ruta_o_archivo).expanduser().resolve()
        if not ruta.exists():
            raise FileNotFoundError(f"No se encontró la imagen en {ruta}")
        imagen = cv2.imread(str(ruta), cv2.IMREAD_COLOR)

    if imagen is None:
        raise ValueError("No se pudo leer la imagen. Verifica el formato o ruta.")

    return imagen


def _extraer_caracteristicas_finas(img_normalizada: np.ndarray) -> np.ndarray:
    """Características finas: bordeado en escala de grises + vector plano."""

    gris = cv2.cvtColor((img_normalizada * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    bordes = cv2.Laplacian(gris, ddepth=cv2.CV_32F)
    return bordes.flatten()


def _extraer_caracteristicas_gruesas(img_normalizada: np.ndarray) -> np.ndarray:
    """Características gruesas: downsample + estadísticos por canal."""

    reducida = cv2.resize(img_normalizada, (32, 32), interpolation=cv2.INTER_AREA)
    plana = reducida.flatten()
    canales = img_normalizada.reshape(-1, 3)
    medias = canales.mean(axis=0)
    desv = canales.std(axis=0)
    return np.concatenate([plana, medias, desv])


def _preprocesar_imagen(ruta_o_archivo: str | Path | Any) -> np.ndarray:
    """Prepara la imagen siguiendo el flujo del entrenamiento EKNN."""

    imagen = _leer_imagen(ruta_o_archivo)
    rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    redimensionada = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
    normalizada = redimensionada.astype(np.float32) / 255.0

    finas = _extraer_caracteristicas_finas(normalizada)
    gruesas = _extraer_caracteristicas_gruesas(normalizada)

    vector = np.concatenate([finas, gruesas]).astype(np.float32)
    return vector


def _obtener_probabilidades(
    modelo: Any, vector: np.ndarray, clases: List[str]
) -> Dict[str, float]:
    """Obtiene probabilidades por clase si el modelo las expone."""

    if hasattr(modelo, "predict_proba"):
        proba = modelo.predict_proba([vector])
        if proba is not None and len(proba):
            return {clase: float(score) for clase, score in zip(clases, proba[0])}

    # Fallback sin probabilidades explícitas: 1.0 para la clase predicha
    pred = modelo.predict([vector])
    pred_clase = pred[0] if len(pred) else None
    return {clase: float(clase == pred_clase) for clase in clases}


def _vecinos_con_etiquetas(
    modelo: Any, vector: np.ndarray
) -> tuple[Optional[List[int]], Optional[List[float]], Optional[List[str]]]:
    """Devuelve índices, distancias y etiquetas de vecinos si están disponibles."""

    if not hasattr(modelo, "kneighbors"):
        return None, None, None

    distancias, vecinos = modelo.kneighbors([vector])
    distancias_list = distancias[0].tolist()
    vecinos_list = vecinos[0].tolist()

    etiquetas_vecinos: Optional[List[str]] = None
    if hasattr(modelo, "_y"):
        try:
            etiquetas_vecinos = [str(modelo._y[idx]) for idx in vecinos_list]  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            etiquetas_vecinos = None

    return vecinos_list, distancias_list, etiquetas_vecinos


def predecir_imagen(ruta_o_archivo: str | Path | Any, modelo=None) -> Dict[str, Any]:
    """Ejecuta la predicción EKNN y devuelve un diccionario JSON-friendly."""

    try:
        modelo = modelo or cargar_modelo()
        vector = _preprocesar_imagen(ruta_o_archivo)

        # Determinar las etiquetas de clase conocidas por el modelo
        clases: List[str] = [str(c) for c in getattr(modelo, "classes_", [])]

        pred = modelo.predict([vector])
        clase_predicha = pred[0] if len(pred) else None

        probabilidades = _obtener_probabilidades(modelo, vector, clases) if clases else None
        confianza = None
        if probabilidades and clase_predicha is not None:
            confianza = float(probabilidades.get(clase_predicha, max(probabilidades.values(), default=0.0)))

        vecinos, distancias, etiquetas_vecinos = _vecinos_con_etiquetas(modelo, vector)

        return {
            "clase_predicha": None if clase_predicha is None else str(clase_predicha),
            "confianza": confianza,
            "probabilidades": probabilidades,
            "vecinos": vecinos,
            "distancias": distancias,
            "etiquetas_vecinos": etiquetas_vecinos,
        }
    except Exception as exc:  # noqa: BLE001
        # Se devuelve el error serializable para manejo en la vista Flask.
        raise ValueError(f"Error al predecir la imagen: {exc}") from exc


__all__ = [
    "cargar_modelo",
    "predecir_imagen",
]
