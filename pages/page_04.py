import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import os
from source.toxicity import text2toxicity, load_model

model_path = "./models/toxicity.pt"
if os.path.exists(model_path):
    pass
else:
    st.error(
        f"Модель {model_path} не найдена. Пожалуйста, убедитесь, что она находится в той же директории, что и приложение Streamlit."
    )
    st.stop()

model, tokenizer = load_model(model_path)

st.header("Определение степени токсичности текста")

st.divider()

st.subheader("Попробуй побить токсичность 9.95💀💀💀")


st.subheader("Выдай самое злое, на что ты способен")

text = st.text_input("Напиши свой самый токсичный комментарий", "Я учусь в Эльбрусе")

st.divider()

toxic_value = text2toxicity(text=text, model=model, tokenizer=tokenizer)[0]*10

if toxic_value <= 2:
    st.markdown(f"""
    ### Да ты еще лапочка🌺
    """)
elif toxic_value <= 5:
    st.markdown("""
    ### Вполне себе обычный комментарий 😊
    """)
elif toxic_value <= 8:
    st.markdown("""
    ### Похоже ты оскорблений знаешь больше, чем обычных слов 😡
    """)
elif toxic_value < 9.95:
    st.markdown("""
    ### Чуточку не дотянул, но для тебя уже есть котел в аду 😈
    """)
else:
    st.markdown("""
    ### Вы просто совершенство, любой спор в интернете за вами 💀💀💀
    """)

st.markdown(f"""
### Твой уровень токсичности {toxic_value:.2f}
""")

