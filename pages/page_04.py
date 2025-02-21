import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import cv2
import numpy as np
from ultralytics import YOLO
import os


model_path = "models/face_yolo11m.pt"
if os.path.exists(model_path):
    model = YOLO(model_path)
else:
    st.error(
        f"Модель {model_path} не найдена. Пожалуйста, убедитесь, что она находится в той же директории, что и приложение Streamlit."
    )
    st.stop()


st.header("Детекция и размытие лиц с помощью YOLO11m")
st.write(" Загрузите изображение или вставьте ссылку для детекции и размытия лиц.")

st.divider()
st.subheader("Вариант 1 : загрузка одной фотографии")
file_1 = st.file_uploader(
    " Загрузите **одно** изображение", type=["jpg", "jpeg", "png"]
)
st.subheader("Вариант 2 : загрузка нескольких фотографий")
file_2 = st.file_uploader(
    " Загрузите **несколько** изображений",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)
st.subheader("Вариант 3 : загрузка фотографии через ссылку ")
url_pic = st.text_input(
    " Вставьте ссылку на изображение",
    "https://thumbs.dreamstime.com/b/красивая-молодая-женщина-сидя-на-стенде-в-парке-города-123244117.jpg",
)

st.divider()
st.subheader("Параметры размытия:")
blur_factor = st.slider(
    " Укажите уровень размытия:", 5, 30, value=15, step=2
)  # Настройка blur_factor

st.divider()


def blur_faces(img, model, blur_factor=15):
    results = model.predict(img)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            face_roi = img[y1:y2, x1:x2]

            ksize = int(max(face_roi.shape) / blur_factor)
            if ksize % 2 == 0:
                ksize += 1
            blurred_face = cv2.GaussianBlur(face_roi, (ksize, ksize), 0)

            img[y1:y2, x1:x2] = blurred_face

    return img


if file_1:
    st.subheader(" Результаты:")
    img = Image.open(file_1)
    cols = st.columns(2)
    with cols[0]:
        st.image(img, caption="Загруженное изображение:")
    with cols[1]:
        img_cv2 = np.array(img)
        img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR)
        blurred_img = blur_faces(img_cv2, model, blur_factor)
        blurred_img_pil = Image.fromarray(cv2.cvtColor(blurred_img, cv2.COLOR_BGR2RGB))
        st.image(blurred_img_pil, caption="Результаты с размытием:")

if file_2:
    st.subheader(" Результаты:")
    cols = st.columns(2)
    images = []
    captions = []

    for i, pic in enumerate(file_2):
        img = Image.open(pic)
        images.append(img)
        captions.append(f"Загруженное изображение {i+1}")

    with cols[0]:
        st.image(images, caption=captions)

    processed_images = []

    for img in images:
        img_cv2 = np.array(img)
        img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR)
        blurred_img = blur_faces(img_cv2, model, blur_factor)
        blurred_img_pil = Image.fromarray(cv2.cvtColor(blurred_img, cv2.COLOR_BGR2RGB))
        processed_images.append(blurred_img_pil)

    with cols[1]:
        st.image(
            processed_images,
            caption=["Размытое изображение " + str(i + 1) for i in range(len(images))],
        )

if url_pic:
    st.subheader(" Результаты:")
    try:
        response = requests.get(url_pic)
        img = Image.open(BytesIO(response.content))
        cols = st.columns(2)
        with cols[0]:
            st.image(img, caption="Загруженное изображение")
        with cols[1]:
            img_cv2 = np.array(img)
            img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR)
            blurred_img = blur_faces(img_cv2, model, blur_factor)
            blurred_img_pil = Image.fromarray(
                cv2.cvtColor(blurred_img, cv2.COLOR_BGR2RGB)
            )
            st.image(blurred_img_pil, caption="Результаты с размытием")

    except Exception as e:
        st.error(f"❌ Ошибка загрузки изображения: {e}")
