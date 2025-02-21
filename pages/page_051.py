import streamlit as st

st.header("Детекция лиц с помощью YOLO11m")
st.header("ℹ️ face_yolo11m.pt: Информация о модели")

st.divider()

epochs = 10 
train_size = 13432
val_size = 3347


st.header(" Основные параметры обучения")
st.write(f" **Число эпох:** `{epochs}`")
st.write(f" **Обучающая выборка:** `{train_size}` изображений")
st.write(f" **Валидационная выборка:** `{val_size}` изображений")
#st.write(f" **Тестовая выборка:** `{test_size}` изображений")
st.markdown(" **Датасет:** [kaggle.com: Набор данных для обнаружения лиц]")

st.divider()

st.header(" Метрики модели")

col1, col2 = st.columns(2)
with col1:
    st.subheader(" F1_curve") 
    st.image('images/F1_curve_face_yolo11m.png', caption='График F1_curve') 

with col2:
    st.subheader(" PR-кривая") 
    st.image('images/PR_curve_face_yolo11m.png', caption='PR-кривая') 


st.subheader(" Precision & Recall")
col1, col2 = st.columns(2)
with col1:
    st.image('images/P_curve_face_yolo11m.png', caption=" Precision") 
with col2:
    st.image('images/R_curve_face_yolo11m.png', caption=" Recall") 

#Матрица ошибок (опционально)
st.subheader(" Confusion Matrix")
col1, col2 = st.columns(2)
with col1:
    st.image('images/confusion_matrix_normalized_face_yolo11m.png', caption="✅ Нормализованная версия") 
with col2:
    st.image('images/confusion_matrix_face_yolo11m.png', caption=" Базовая версия") 

st.divider()

st.header(" Примеры работы модели")

st.image('images/face_yolo11m_1.jpg', caption='Результат Детекция лиц') 
st.image('images/face_yolo11m_11.jpg', caption='Результат Детекция лиц') 

st.image('images/face_yolo11m_22.jpg', caption='Результат Детекция лиц') 
st.image('images/face_yolo11m_3.jpg', caption='Результат Детекция лиц')

st.image('images/face_yolo11m_88.jpg', caption='Результат с размытием 8') 
st.image('images/face_yolo11m_55.jpg', caption='Результат с размытием 5') 
