"""Flask API for KNN image inference."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

from inference import (
    load_trained_model,
    preprocess_image_file,
    predict_classes,
)

BASE_DIR = Path(__file__).parent
DEFAULT_MODEL_PATH = BASE_DIR / "modelos" / "modelo_final_eknn.pkl"

def _parse_target_size(value: str | None) -> tuple[int, int]:
    if not value:
        return (224, 224)
    try:
        width, height = value.split(",")
        return (int(width), int(height))
    except (ValueError, TypeError):
        raise ValueError("TARGET_SIZE must be formatted as 'width,height'")

def _load_class_labels(value: str | None, model_labels: List[str] | None, num_classes: int) -> List[str]:
    if value:
        labels = [label.strip() for label in value.split(",") if label.strip()]
        if labels and len(labels) == num_classes:
            return labels
    if model_labels and len(model_labels) == num_classes:
        return list(model_labels)
    return [f"class_{i}" for i in range(num_classes)]


def create_app() -> Flask:
    model_path = Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)).expanduser()
    target_size = _parse_target_size(os.getenv("TARGET_SIZE"))

    app = Flask(__name__)
    CORS(app)

    model = load_trained_model(model_path)
    model_labels = getattr(model, "classes_", None)

    @app.get("/health")
    def health_check():
        return jsonify({"status": "ok"})

    @app.post("/predict")
    def predict():
        if "image" not in request.files:
            return jsonify({"error": "No se adjuntó el archivo 'image'."}), 400

        file_storage = request.files["image"]
        if file_storage.filename == "":
            return jsonify({"error": "El archivo subido está vacío."}), 400

        try:
            processed = preprocess_image_file(file_storage, target_size)
            predictions = predict_classes(model, [processed])
            if predictions.size == 0:
                raise ValueError("El modelo no devolvió predicciones.")

            probabilities = predictions[0]
            num_classes = probabilities.shape[-1]
            class_labels = _load_class_labels(os.getenv("CLASS_LABELS"), model_labels, num_classes)

            best_index = int(np.argmax(probabilities))
            best_confidence = float(probabilities[best_index])
            best_label = class_labels[best_index]

            probs_by_class = {
                label: round(float(prob) * 100, 2)
                for label, prob in zip(class_labels, probabilities)
            }

            return jsonify(
                {
                    "prediccion": best_label,
                    "confianza": round(best_confidence * 100, 2),
                    "probabilidades": probs_by_class,
                }
            )
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": str(exc)}), 500

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)
