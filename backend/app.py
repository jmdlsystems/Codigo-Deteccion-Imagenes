"""Flask API para exponer el modelo EKNN via HTTP."""
from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS

from inference import cargar_modelo, predecir_imagen

BASE_DIR = Path(__file__).parent
DEFAULT_MODEL_PATH = BASE_DIR / "modelos" / "modelo_final_eknn.pkl"


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    model_path = Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)).expanduser()
    modelo = cargar_modelo(model_path)

    @app.get("/health")
    def health_check():
        return jsonify({"status": "ok"})

    @app.post("/predict")
    def predict():
        if "image" not in request.files:
            return jsonify({"error": "No se adjuntó el archivo 'image'."}), 400

        archivo = request.files["image"]
        if archivo.filename == "":
            return jsonify({"error": "El archivo subido está vacío."}), 400

        try:
            resultado = predecir_imagen(archivo, modelo)
            return jsonify(resultado)
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": str(exc)}), 500

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)
