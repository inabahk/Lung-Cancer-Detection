import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# ✅ MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Lung Cancer Detection",
    layout="centered"
)

# -------------------------------
# CONFIG
# -------------------------------
MODEL_PATH = "lung_cancer_xception.keras"
IMG_SIZE = 299

# ⚠️ IMPORTANT: Update this if your class order is different
CLASS_NAMES = ["Bengin cases", "Malignant cases", "Normal cases"]

# -------------------------------
# LOAD MODEL (cached)
# -------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found: {MODEL_PATH}")
        st.stop()
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# -------------------------------
# UI
# -------------------------------
st.title("🫁 Lung Cancer Detection")
st.write("Upload a lung CT scan image to predict cancer type.")

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------
# PREDICTION FUNCTION
# -------------------------------
def preprocess_image(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0

    # Ensure 3 channels
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -------------------------------
# MAIN LOGIC
# -------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Predict"):
        with st.spinner("Analyzing..."):
            processed_img = preprocess_image(image)
            predictions = model.predict(processed_img)

            predicted_class = CLASS_NAMES[np.argmax(predictions)]
            confidence = np.max(predictions) * 100

        # -------------------------------
        # OUTPUT
        # -------------------------------
        st.success(f"**Prediction:** {predicted_class}")
        st.info(f"**Confidence:** {confidence:.2f}%")

        st.subheader("📊 Class Probabilities:")
        for i, cls in enumerate(CLASS_NAMES):
            st.write(f"{cls}: {predictions[0][i]*100:.2f}%")