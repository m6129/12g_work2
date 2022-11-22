import io # обязательные библиотеки для stremlit
import streamlit as st # # обязательные библиотеки для stremlit
from PIL import Image # библиотека для загрузки изображений
#import torch
#from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from transformers import pipeline

image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

#print("Эта модель развернута группой студентов УрФу")

@st.cache(allow_output_mutation=True)
def load_model():
    return image_to_text

def load_image():
    uploaded_file = st.file_uploader(label='Выберите изображение для распознования') # загрузчик файлов
    if uploaded_file is not None: # если пользователь загрузил файл
        image_data = uploaded_file.getvalue() # то мы его читаем
        st.image(image_data) # преобразуем с помощью средств stremlit
        return Image.open(io.BytesIO(image_data))# возвращаем это изображение
    else:
        return None
    
st.title('Классификация изображений')
img = load_image() # вызываем функцию

#result = st.button('Распознать изображение')
#if result:
#    x = preprocess_image(img)
#    preds = model.predict(x)
#    st.write('**Результаты распознавания:**')
#    print_predictions(preds)
