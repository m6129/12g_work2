import io # обязательные библиотеки для stremlit
import streamlit as st # # обязательные библиотеки для stremlit
from PIL import Image # библиотека для загрузки изображений
#import torch
#from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from transformers import pipeline

#image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
@st.cache(allow_output_mutation=True)
def load_model():
    return image_to_text
image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

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

if result: #после нажатия на которую будет запущен алгоритм...
    x = preprocess_image(img)
    preds = model(x)
    st.write('**Результаты распознавания:**')
    print_predictions(preds)
      #preds = model.predict(x)
