import os
import json
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import gdown

# Setup Layout
st.set_page_config(page_title="Agriscan", layout="wide")

# URL Google Drive (ganti dengan ID file model yang benar)
FILE_ID = "1XyzABC1234h5EXAMPLE"  # Ganti dengan ID file dari Google Drive
MODEL_PATH = "model-prediksi-penyakit-tanaman.h5"
MODEL_URL = f"https://drive.google.com/uc?id={FILE_ID}"

# Cek dan unduh model jika belum ada
if not os.path.exists(MODEL_PATH):
    with st.spinner("Mengunduh model dari Google Drive..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Load class indices
class_indices_path = "class_indices.json"
with open(class_indices_path, "r") as f:
    class_indices = json.load(f)
    class_indices = {int(k): v for k, v in class_indices.items()}

# Daftar tanaman buah yang bisa diprediksi
plant_list = [
    "🍏 Apple", "🫐 Blueberry", "🍒 Cherry", "🍇 Grape",
    "🍑 Peach", "🍓 Raspberry", "🍓 Strawberry", "🍅 Tomato"
]

# Fungsi untuk memproses gambar
def preprocess_image(image_file, target_size=(224, 224)):
    img = Image.open(image_file)
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0  # Normalisasi
    img_array = np.expand_dims(img_array, axis=0)  # Tambahkan batch dimensi
    return img_array

# Fungsi untuk prediksi
def predict_image_class(model, image_file, class_indices):
    preprocessed_img = preprocess_image(image_file)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices.get(predicted_class_index, "Unknown")
    confidence = np.max(predictions) * 100  # Ambil confidence level
    return predicted_class_name, confidence

# UI Streamlit
st.title('🌱 Agriscan')
st.write("🚀 **Website Prediksi Penyakit pada Tanaman Buah Sub Tropis**")

# Menampilkan daftar tanaman buah dalam expander
with st.expander("🌿 Daftar Tanaman Buah yang Dapat Diprediksi:"):
    for plant in plant_list:
        st.markdown(f"<p style='font-size: 13px;'>{plant}</p>", unsafe_allow_html=True)

# File Uploader
uploaded_image = st.file_uploader("📤 Unggah gambar daun buah untuk mengetahui penyakitnya!", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    st.markdown("---")
    image = Image.open(uploaded_image)

    col1, col2 = st.columns([1, 2])
    with col1:
        resized_img = image.resize((200, 200))
        st.image(resized_img, caption="📷 Gambar yang Diupload", use_container_width=True)
    
    with col2:
        st.markdown("### 🔍 Analisis Model")
        if st.button('🔎 Prediksi Sekarang'):
            with st.spinner("🔄 Sedang menganalisis gambar..."):
                prediction, confidence = predict_image_class(model, uploaded_image, class_indices)
            
            st.markdown(f"<h3 style='color: #5fff3f;'>✅ Hasil Prediksi: {prediction}</h3>", unsafe_allow_html=True)
            st.progress(confidence / 100)
            st.write(f"📊 **Kepercayaan Model: {confidence:.2f}%**")
            
            if confidence < 50:
                st.warning("⚠️ Model kurang yakin dengan prediksi ini. Coba unggah gambar lain dengan kualitas lebih baik.")
