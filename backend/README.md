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

## Ejemplo de uso

```python
from backend.inference import (
    load_trained_model,
    preprocess_image,
    predict_classes,
)

model = load_trained_model("backend/modelos/mejor_modelo_maiz_efficientnet.keras.h5")
image = preprocess_image("ruta/a/imagen.jpg", target_size=(224, 224))
predictions = predict_classes(model, [image])
print(predictions)
```
