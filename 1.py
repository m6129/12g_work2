import io
import streamlit as st
from PIL import Image
import torch
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from transformers import pipeline

def load_image():
    uploaded_file = st.file_uploader(label='Выберите изображение для распознования') # загрузчик файлов
    if uploaded_file is not None: # если пользователь загрузил файл
        image_data = uploaded_file.getvalue() # то мы его читаем
        st.image(image_data) # преобразуем с помощью средств stremlit
        return Image.open(io.BytesIO(image_data))# возвращаем это изображение
    else:
        return None

st.title('Классификация изображений')
img = load_image()




image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

image_to_text("https://ankur3107.github.io/assets/images/image-captioning-example.png")
