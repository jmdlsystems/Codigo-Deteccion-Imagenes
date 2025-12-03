"""Inference helpers for the scikit-learn KNN model on Python 3.13.

The backend now uses the documented ``modelo_final_eknn.pkl`` artifact saved in
``backend/modelos/``. This module provides a thin wrapper around the pickled
model plus some Pillow-based preprocessing to convert uploaded images into the
flat numeric arrays expected by the KNN classifier.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import joblib
import numpy as np
import PIL.Image as PILImage


def load_trained_model(model_path: str | Path):
    """Load the pickled KNN model from disk."""

    resolved_path = Path(model_path).expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"No model file found at {resolved_path}")

    return joblib.load(resolved_path)


def preprocess_image(image_path: str | Path, target_size: Tuple[int, int]) -> np.ndarray:
    """Load and normalize an image from disk into a flat feature vector."""

    resolved_path = Path(image_path).expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"No image file found at {resolved_path}")

    with resolved_path.open("rb") as stream:
        return preprocess_image_file(stream, target_size)


def preprocess_image_file(file_like, target_size: Tuple[int, int]) -> np.ndarray:
    """Normalize an uploaded image stream for inference.

    Accepts file-like objects (e.g., Flask's ``FileStorage``) so the backend can
    process uploads without writing them to disk.
    """

    stream = getattr(file_like, "stream", file_like)

    if hasattr(stream, "seek"):
        stream.seek(0)

    img = PILImage.open(stream).convert("RGB")
    img = img.resize(target_size)

    array = np.asarray(img, dtype=np.float32) / 255.0
    flat = array.flatten()
    return flat


def predict_classes(model, processed_images: Iterable[np.ndarray]) -> np.ndarray:
    """Run model predictions over a batch of preprocessed images."""

    batch = np.vstack([np.array(img, dtype=np.float32) for img in processed_images])

    if hasattr(model, "predict_proba"):
        return model.predict_proba(batch)

    # Fallback: derive pseudo-probabilities from distances if available.
    if hasattr(model, "kneighbors"):
        distances, indices = model.kneighbors(batch)
        probs = 1 / (distances + 1e-8)
        probs = probs / probs.sum(axis=1, keepdims=True)
        # Map neighbor indices to classes when possible.
        if hasattr(model, "classes_"):
            num_classes = len(model.classes_)
            prob_matrix = np.zeros((batch.shape[0], num_classes), dtype=np.float32)
            for row, (neighbor_indices, weight_row) in enumerate(zip(indices, probs)):
                for neighbor_idx, weight in zip(neighbor_indices, weight_row):
                    class_idx = neighbor_idx % num_classes
                    prob_matrix[row, class_idx] += weight
            prob_matrix = prob_matrix / prob_matrix.sum(axis=1, keepdims=True)
            return prob_matrix
        return probs

    # Last resort: use predicted labels with uniform confidence.
    predictions = model.predict(batch)
    classes = getattr(model, "classes_", np.unique(predictions))
    num_classes = len(classes)
    prob_matrix = np.full((batch.shape[0], num_classes), 1 / num_classes, dtype=np.float32)
    for row, label in enumerate(predictions):
        if label in classes:
            idx = list(classes).index(label)
            prob_matrix[row] = 0
            prob_matrix[row, idx] = 1.0
    return prob_matrix


__all__ = [
    "load_trained_model",
    "preprocess_image",
    "preprocess_image_file",
    "predict_classes",
]
