import streamlit as st

st.header("Определение токсичности текста")
st.header("ℹ️ rubert-tiny-toxicity: Информация о модели")

st.divider()

epochs = 50
train_size = 11531
val_size = 2882


st.header(" Основные параметры обучения")
st.write(f" **Число эпох:** `{epochs}`")
st.write(f" **Обучающая выборка:** `{train_size}` комментариев")
st.write(f" **Валидационная выборка:** `{val_size}` комментариев")
#st.write(f" **Тестовая выборка:** `{test_size}` изображений")
st.markdown(" **Датасет:** Токсичные комментарии")

st.divider()

st.header(" Метрики модели")

st.subheader(" Precision & Recall")
col1, col2 = st.columns(2)
with col1:
    st.image('images/precision_toxicity.png', caption=" Precision")
with col2:
    st.image('images/recall_toxicity.png', caption=" Recall")

st.subheader(" Loss & Accuracy")
col1, col2 = st.columns(2)
with col1:
    st.image('images/loss_toxicity.png', caption=" Precision")
with col2:
    st.image('images/accuracy_toxic.png', caption=" Recall")

st.divider()

