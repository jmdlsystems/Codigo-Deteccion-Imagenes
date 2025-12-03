"""Inference helpers using TensorFlow on Python 3.13.

This module keeps the TensorFlow usage limited to the v2 public API so it
remains forward-compatible with Python 3.13 wheels. Loading and predicting are
wrapped in small helper functions so callers do not need to manage sessions or
low-level graph constructs.
"""

from __future__ import annotations

import io
from pathlib import Path
from PIL import Image
from typing import Iterable, Tuple

import numpy as np
import tensorflow as tf


def load_trained_model(model_path: str | Path) -> tf.keras.Model:
    resolved_path = Path(model_path).expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"No model file found at {resolved_path}")

    return tf.keras.models.load_model(resolved_path)


def preprocess_image(image_path: str | Path, target_size: Tuple[int, int]) -> np.ndarray:
    resolved_path = Path(image_path).expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"No image file found at {resolved_path}")

    with resolved_path.open("rb") as stream:
        return preprocess_image_file(stream, target_size)


def preprocess_image_file(file_storage, target_size):
    """
    Convierte una imagen subida vía Flask (FileStorage) o stream
    en un tensor numpy listo para el modelo.
    """
    image_bytes = file_storage.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(target_size)

    img_array = np.array(img).astype("float32") / 255.0
    return img_array


def predict_classes(
    model: tf.keras.Model,
    processed_images: Iterable[np.ndarray]
) -> np.ndarray:
    """
    Corre predicciones creando un batch correcto para EfficientNet:
    (N, 224, 224, 3)
    """

    # Convertir cada imagen en un array (224,224,3)
    imgs = [np.array(img) for img in processed_images]

    # Crear batch (N,224,224,3)
    batch = np.stack(imgs, axis=0)

    return model.predict(batch, verbose=0)


__all__ = [
    "load_trained_model",
    "preprocess_image",
    "preprocess_image_file",
    "predict_classes",
]
