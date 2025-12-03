# Backend (Python 3.13 + modelo_final_eknn.pkl)

El backend expone una API Flask que carga el modelo documentado
`backend/modelos/modelo_final_eknn.pkl` y permite subir imágenes desde el
frontend para medir la confianza del clasificador KNN.

## Requisitos

1. Python 3.13
2. Dependencias de runtime:
   ```bash
   python -m pip install --upgrade pip
   pip install -r backend/requirements.txt
   ```

Las dependencias usan scikit-learn y Pillow, por lo que no necesitas instalar
TensorFlow para correr la API con el modelo KNN en CPython 3.13.

## API Flask para conectar con el frontend

1. Configura variables opcionales (por defecto se usa el modelo EfficientNet en
   `backend/modelos/` y tamaño de entrada 224x224):
   ```bash
   export MODEL_PATH="backend/modelos/modelo_final_eknn.pkl"
   export TARGET_SIZE="224,224"  # ancho,alto
   export CLASS_LABELS="label_a,label_b,label_c"  # opcional, coincide con el orden de salida del modelo
   ```

2. Arranca el servidor Flask:
   ```bash
   cd backend
   python app.py
   ```

3. Envía una petición POST desde el frontend o cURL:
   ```bash
   curl -X POST \
        -F "image=@/ruta/a/hoja.jpg" \
        http://localhost:5000/predict
   ```

La respuesta incluye la clase con mayor probabilidad, la confianza en %, y las
probabilidades por clase.
