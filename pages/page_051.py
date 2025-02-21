import streamlit as st

st.header("Генерация текста с помощью GPT3-medium-based-on-GPT2")
st.header("ℹ️ Информация о модели")

st.divider()

epochs = 8.3
train_size = 8341   # строка
word_size = 44264  # слова
timer = 4.5


st.header(" Основные параметры обучения")
st.write(f'**Обучала на текстовом документе с текстами песен**')
st.markdown(
    f"<h3 style='text-align: center; font-size:24px;'>Число эпох: <span style='color:#7c2ec9;'>{epochs}</span></h3>", 
    unsafe_allow_html=True
)
st.markdown(
    f"<h3 style='text-align: center; font-size:24px;'>Число строк: <span style='color:#7c2ec9;'>{train_size}</span></h3>", 
    unsafe_allow_html=True
)
st.markdown(
    f"<h3 style='text-align: center; font-size:24px;'>Число слов: <span style='color:#7c2ec9;'>{word_size}</span></h3>", 
    unsafe_allow_html=True
)
st.markdown(
    f"<h3 style='text-align: center; font-size:24px;'>Затраченное время: <span style='color:#7c2ec9;'>{timer} часа</span></h3>", 
    unsafe_allow_html=True
)
st.markdown(" **Модель на hugging-face:** [вот сюда жми ага](https://huggingface.co/RenaTheDv/rugpt-medium-based-on-gpt2-test1)")

st.divider()

st.header(" Немного про модель ")

st.write('Показатели при обучении модели (не в эпохах, а в шагах)')
st.image('images/model_metric.png')

st.header('На чем обучали модель?')

col1, col2 = st.columns(2)
with col1: 
    st.subheader('Примеры текста:')
    st.image('images/atl_example.png', width=400, caption='') 

with col2:
    st.markdown(
        "<div style='display: flex; align-items: center; height: 100%;'>"
        "<p style='padding-top: 85px; font-size: 15px; font-weight: bold;'>Было принято решение начать с текстов такого исполнителя, как ATL, поскольку во многих его песнях создавался некий свой мир, который всегда был мне интересен.</p>"
        "</div>",
        unsafe_allow_html=True
    )

col3, col4 = st.columns(2)
with col3:
    st.markdown(
        "<div style='display: flex; align-items: center; height: 100%;'>"
        "<p style='padding-top: 125px; font-size: 15px; font-weight: bold;'>Пример был взять из данного альбома. Помимо него, взяты были ВСЕ песни исполнителя (парсинг сайта с текстами - это нечто)...</p>"
        "</div>",
        unsafe_allow_html=True
    )
with col4:
    st.image('images/atl_image.png')

col5, col6 = st.columns(2)
with col5: 
    st.image('images/boul_example.png', width=400, caption='') 

with col6:
    st.markdown(
        "<div style='display: flex; align-items: center; height: 100%;'>"
        "<p style='padding-top: 25px; font-size: 15px; font-weight: bold;'>164 песни ATL - именно столько песен у исполнителя без ремиксов - необходимо было дополнить. Был выбран, в том числе, и исполнитель Boulevard Depo - c его мрачными текстами.</p>"
        "</div>",
        unsafe_allow_html=True
    )

col7, col8 = st.columns(2)
with col7:
    st.markdown(
        "<div style='display: flex; align-items: center; height: 100%;'>"
        "<p style='padding-top: 125px; font-size: 15px; font-weight: bold;'>Пример был взять из данного альбома. Помимо него, было взято еще около 15 песен...</p>"
        "</div>",
        unsafe_allow_html=True
    )
with col8:
    st.image('images/boul_image.jpg')

col9, col10 = st.columns(2)
with col9: 
    st.image('images/miya_example.png', width=400, caption='') 

with col10:
    st.markdown(
        "<div style='display: flex; align-items: center; height: 100%;'>"
        "<p style='padding-top: 105px; font-size: 15px; font-weight: bold;'>Решила разбавить серьезные смыслы кальянным репом - Мияги и Эндшпиль мне в помощь.</p>"
        "</div>",
        unsafe_allow_html=True
    )

col11, col12 = st.columns(2)
with col11:
    st.markdown(
        "<div style='display: flex; align-items: center; height: 100%;'>"
        "<p style='padding-top: 125px; font-size: 15px; font-weight: bold;'>Пример был взять из данного альбома. Помимо него, было взято еще около 10 кальянных песен...</p>"
        "</div>",
        unsafe_allow_html=True
    )
with col12:
    st.image('images/miya_image.jpg')

st.header('Общие выводы')
st.write('Модель обучалась очень долго - большое количество времени ушло на поиск возможности обучать модель в течение 4 часов. При первом обучении Google Colab просто вылетел на 1,5 часах :)')
st.write('Модели явно не хватает больше времени на обучение и больше текстов - для этого нужно немного больше. Поэтому она пока не может писать полноценные тексты с адекватным смыслом... Но она уже пишет лучше половины наших исполнителей.')
st.header('Идеи для обновления модели')
st.write('Сделать телеграм-бота, который принимает фразу и выдает текст. Поработать с обучением - сделать больший текстовый документ (минимум в два раза), найти больше мощностей и обучить повторно.')