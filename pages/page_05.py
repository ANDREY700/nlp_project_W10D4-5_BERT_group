import streamlit as st
import requests
from PIL import Image
#from models.model import predict
from io import BytesIO
import cv2
import numpy as np
import ultralytics
from ultralytics import YOLO


st.title("🔍 YOLO11: Детекция ветрогенераторов")
st.write("🖼 Загрузите изображение или вставьте ссылку для детекции объектов.")

st.divider()

file_1 = st.file_uploader("📥 Загрузите **одно** изображение", type=["jpg", "jpeg", "png"])
file_2 = st.file_uploader("📂 Загрузите **несколько** изображений", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
url_pic = st.text_input("🌐 Вставьте ссылку на изображение")

conf = st.slider("🎯 Укажите confidence:", 0.0, 1.0, value=0.5)

st.divider()



def predict(img, conf):
    model = YOLO('models/best-5.pt')
    results = model(img, conf=conf)
    return results

if file_1:
    st.subheader('🔍 Результаты:')
    img = Image.open(file_1)
    cols = st.columns(2)
    if conf:
        with cols[0]:
            st.image(img, caption='Загруженное изображение:')
        with cols[1]:
            
            img_cv2 = np.array(img)
            img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR)
            results = predict(img_cv2, conf)
            result_img = results[0].plot()
            result_pil = Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))

            st.image(result_pil, caption="Результаты детекции:")


if file_2:
    st.subheader('🔍 Результаты:')
    cols = st.columns(2)  # Два столбца для показа изображений
    images = []
    captions = []
    
    for i, pic in enumerate(file_2):
        img = Image.open(pic)
        images.append(img)
        captions.append(f"Загруженное изображение {i+1}")

    with cols[0]:
        st.image(images, caption=captions)

    if conf:
        processed_images = []
        
        for img in images:
            img_cv2 = np.array(img)
            img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR)
            results = predict(img_cv2, conf)
            result_img = results[0].plot()
            result_pil = Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
            processed_images.append(result_pil)

        with cols[1]:
            st.image(processed_images, caption=["Детектированное изображение " + str(i+1) for i in range(len(images))])

if url_pic:
    st.subheader('🔍 Результаты:')
    try:
        response = requests.get(url_pic)
        img = Image.open(BytesIO(response.content))
        if conf:
            cols = st.columns(2)
            with cols[0]:
                st.image(img, caption="Загруженное изображение")
            with cols[1]:
                img_cv2 = np.array(img)
                img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR)
                results = predict(img_cv2, conf)
                result_img = results[0].plot()
                result_pil = Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))

                st.image(result_pil, caption="Результаты детекции")

    except Exception as e:
        st.error(f"❌ Ошибка загрузки изображения: {e}")
