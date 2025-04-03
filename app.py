import os
import json
import numpy as np
import streamlit as st
from PIL import Image
import pandas as pd
import requests
from io import BytesIO

# Setup Layout
st.set_page_config(page_title="Agriscan", layout="wide")

# Menambahkan gambar banner tanaman di bagian atas
banner_image_url = "https://barossa.coop/wp-content/uploads/2022/07/indoor-2-1080-500-px-1080-400-px-1080-300-px-2.png"  # Gantilah dengan path gambar banner Anda
response = requests.get(banner_image_url)  # Mengunduh gambar dari URL
banner_image = Image.open(BytesIO(response.content))  # Membuka gambar yang telah diunduh

# Crop gambar (ambil lebar penuh dan sesuaikan tinggi)
width, height = banner_image.size
crop_height = 150  # Sesuaikan tinggi crop sesuai keinginan Anda
cropped_banner_image = banner_image.crop((0, 0, width, crop_height))  # Crop bagian atas dengan tinggi yang ditentukan

# Tampilkan gambar banner yang telah dicrop
st.image(cropped_banner_image, use_container_width=True)

# Mendapatkan direktori kerja
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, "model-prediksi-penyakit-tanaman.h5")
class_indices_path = os.path.join(working_dir, "class_indices.json")

# Load model
model = tf.keras.models.load_model(model_path)

# Load class indices
with open(class_indices_path, "r") as f:
    class_indices = json.load(f)
    class_indices = {int(k): v for k, v in class_indices.items()}  # Ubah key ke int

# Daftar tanaman buah yang bisa diprediksi
plant_list = [
    "🍏 Apple",
    "🫐 Blueberry",
    "🍒 Cherry",
    "🍇 Grape",
    "🍑 Peach",
    "🍓 Raspberry",
    "🍓 Strawberry",
    "🍅 Tomato"
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
            st.progress(confidence / 100)  # Menampilkan progress bar untuk confidence level
            st.write(f"📊 **Kepercayaan Model: {confidence:.2f}%**")

            if confidence < 50:
                st.warning("⚠️ Model kurang yakin dengan prediksi ini. Coba unggah gambar lain dengan kualitas lebih baik.")
