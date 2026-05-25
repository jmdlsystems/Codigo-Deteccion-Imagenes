import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, applications
from PIL import Image
import joblib

# ----------------------------------------------------------
# CONFIGURACIÓN PRINCIPAL
# ----------------------------------------------------------
IMG_SIZE = 224

st.set_page_config(page_title="Detección de Enfermedades del Maíz", layout="centered")

st.title("🌽 Detección de Enfermedades del Maíz con EKNN + DenseNet121")
st.write("Sube una imagen de una hoja de maíz para obtener la predicción del modelo.")

# ----------------------------------------------------------
# Cargar modelo EKNN + clase
# ----------------------------------------------------------

@st.cache_resource
def load_knn():
    return joblib.load("modelo_final_eknn.pkl")  # <-- Este archivo debe existir

@st.cache_resource
def load_class_names():
    # Modifica con tus clases reales
    return ['mancha_gris_foliar', 'roya_comun', 'sana', 'tizon_foliar']

knn_model = load_knn()
class_names = load_class_names()

# ----------------------------------------------------------
# Cargar extractor de características basado en DenseNet121
# ----------------------------------------------------------

@st.cache_resource
def load_extractor():
    base_model = applications.DenseNet121(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3),
        pooling='avg'
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(224, 224, 3))
    x = layers.Rescaling(255.0)(inputs)
    x = applications.densenet.preprocess_input(x)
    feats = base_model(x, training=False)

    model = models.Model(inputs, feats)
    return model

extractor_model = load_extractor()

# ----------------------------------------------------------
# Función para procesar imágenes (RGB / RGBA)
# ----------------------------------------------------------

def load_and_prepare_image(img: Image.Image):
    """
    Soporta imágenes con 3 o 4 canales.
    Convierte todo a RGB, redimensiona y normaliza.
    """

    # Convertir siempre a RGB (si viene RGBA la convierte bien)
    img = img.convert("RGB")

    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1,224,224,3)
    return img_array

# ----------------------------------------------------------
# Subir archivo
# ----------------------------------------------------------
uploaded_file = st.file_uploader("📤 Sube una imagen (PNG, JPG, WEBP, TIFF...)", type=["png", "jpg", "jpeg", "webp", "bmp", "tiff"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼 Imagen cargada", use_column_width=True)

    st.write("Procesando imagen...")

    # Procesamiento seguro
    try:
        img_array = load_and_prepare_image(image)

        # Extraer características
        features = extractor_model.predict(img_array, verbose=0)

        # Predicción EKNN
        pred_label = knn_model.predict(features)[0]
        pred_proba = knn_model.predict_proba(features)[0]

        st.markdown("---")
        st.subheader("Resultado")

        st.write(f"**Predicción:** {class_names[pred_label]}")
        st.write("**Probabilidades por clase:**")

        # Mostrar probabilidades
        for cls, p in zip(class_names, pred_proba):
            st.write(f"- **{cls}:** {p*100:.2f}%")

    except Exception as e:
        st.error(f"Error procesando la imagen: {e}")

