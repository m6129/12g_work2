import io # обязательные библиотеки для stremlit
import streamlit as st # # обязательные библиотеки для stremlit
from PIL import Image # библиотека для загрузки изображений
#import torch
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from transformers import pipeline

@st.cache(allow_output_mutation=True)
def load_model():
    m = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
    return m


def load_image():
    uploaded_file = st.file_uploader(label='Загрузите пожалуйста изображение') # загрузчик файлов
    if uploaded_file is not None: # если пользователь загрузил файл
        image_data = uploaded_file.getvalue() # то мы его читаем
        st.image(image_data) # преобразуем с помощью средств stremlit
        return Image.open(io.BytesIO(image_data))# возвращаем это изображение
    else:
        return None
    
st.title('Классификация изображений')
img = load_image() # вызываем функцию
result = st.button('Распознать изображение')# вставляем кнопку
mod = load_model()
st.write('**Успешно:**')
if result: #после нажатия на которую будет запущен алгоритм...
    st.write('**Результаты распознавания:**')
    m(img)
    mod(img)
