"""Inference helpers using TensorFlow on Python 3.13.

This module keeps the TensorFlow usage limited to the v2 public API so it
remains forward-compatible with Python 3.13 wheels. Loading and predicting are
wrapped in small helper functions so callers do not need to manage sessions or
low-level graph constructs.
"""

from __future__ import annotations

import io
from pathlib import Path
from tkinter import Image
from typing import Iterable, Tuple

import numpy as np
import tensorflow as tf


def load_trained_model(model_path: str | Path) -> tf.keras.Model:
    """Load a saved TensorFlow model from disk.

    Parameters
    ----------
    model_path:
        Path to a `.h5` or `.keras` model file. The function relies on the
        TensorFlow v2 loader, which is supported on Python 3.13.

    Returns
    -------
    tf.keras.Model
        A compiled Keras model ready for inference.
    """

    resolved_path = Path(model_path).expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"No model file found at {resolved_path}")

    # Using the standard Keras loader keeps compatibility with TF 2.x wheels
    # built for CPython 3.13.
    return tf.keras.models.load_model(resolved_path)


def preprocess_image(image_path: str | Path, target_size: Tuple[int, int]) -> np.ndarray:
    """Load and normalize an image for inference.

    The preprocessing relies on TensorFlow's high-level utilities available in
    the v2 API and avoids deprecated symbols removed in recent releases.
    """

    resolved_path = Path(image_path).expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"No image file found at {resolved_path}")

    with resolved_path.open("rb") as stream:
        return preprocess_image_file(stream, target_size)



def preprocess_image_file(file_storage, target_size):
    """
    Convierte una imagen subida vía Flask (FileStorage)
    en un tensor numpy listo para el modelo.
    """
    # Leer bytes del archivo subido
    image_bytes = file_storage.read()

    # Convertir bytes → imagen PIL
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Redimensionar
    img = img.resize(target_size)

    # Convertir a arreglo numpy normalizado
    img_array = np.array(img).astype("float32") / 255.0

    return img_array
def predict_classes(
    model: tf.keras.Model, processed_images: Iterable[np.ndarray]
) -> np.ndarray:
    """Run model predictions over a batch of preprocessed images.

    Parameters
    ----------
    model:
        A TensorFlow Keras model returned by :func:`load_trained_model`.
    processed_images:
        Iterable of images returned by :func:`preprocess_image`.

    Returns
    -------
    np.ndarray
        Model predictions stacked into a NumPy array.
    """

    # Convert to a single NumPy batch for predict to avoid eager execution
    # incompatibilities on older graphs.
    batch = np.vstack(list(processed_images))
    return model.predict(batch, verbose=0)


__all__ = [
    "load_trained_model",
    "preprocess_image",
    "preprocess_image_file",
    "predict_classes",
]
