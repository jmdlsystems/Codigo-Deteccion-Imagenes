# Backend (Python 3.13 + TensorFlow)

Este proyecto no incluía código de backend, pero los modelos guardados en
`backend/modelos/` pueden cargarse para inferencia con Python 3.13 usando
TensorFlow. Usa las utilidades del módulo `inference.py` para trabajar con los
modelos Keras (`.h5` o `.keras`).

## Requisitos

1. Python 3.13
2. Dependencias de runtime:
   ```bash
   python -m pip install --upgrade pip
   pip install -r backend/requirements.txt
   ```

Las versiones indicadas usan la API pública de TensorFlow 2.x, por lo que no
requieren `tf.compat.v1` y funcionan con las ruedas distribuidas para CPython
3.13.

## API Flask para conectar con el frontend

1. Configura variables opcionales (por defecto se usa el modelo EfficientNet en
   `backend/modelos/` y tamaño de entrada 224x224):
   ```bash
   export MODEL_PATH="backend/modelos/mejor_modelo_maiz_efficientnet.keras.h5"
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
