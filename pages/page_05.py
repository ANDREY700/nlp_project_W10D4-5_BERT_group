import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import random
import time


# подгружаем модель
model_name = "RenaTheDv/rugpt-medium-based-on-gpt2-test1"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)


# функция для генерации этого бреда))
def generate_text(prompt, max_length=100, temperature=0.9, top_k=50, top_p=0.9, num_return_sequences=1):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(
        input_ids,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        num_return_sequences=num_return_sequences,
        do_sample=True,
        repetition_penalty=1.3,
        no_repeat_ngram_size=3
    )
    return [tokenizer.decode(out, skip_special_tokens=True) for out in output]


# для разного цвета текста
colors = ['#4287f5','#6ed134','#d17b34','#9534d1','#bd2b79','#bd332b','#bdba2b','#2bbda2','#bd2bac','#502bbd']
random_color = random.choice(colors)

start_time = time.time()

st.title("ruGPT3-medium-based-on-GPT2 от sberbank.ai")
st.write("Генерация текста с помощью предобученной модели на русском языке.")

st.divider()

prompt = st.text_area('Введите начальную фразу', 'Ну допустим йоу')

st.markdown("""
    <style>
        /* Изменение цвета фона слайдера */
        div[data-baseweb="slider"] > div {
            background: linear-gradient(90deg, #27219c, #219c6c); /* Градиент */
            border-radius: 10px;  /* Скругление */
        }

        /* Уменьшение высоты слайдера */
        div[data-baseweb="slider"] {
            height: 6px !important;
        }

        /* Стилизация ползунка */
        div[data-testid="stTickBarMin"], 
        div[data-testid="stTickBarMax"] {
            color: #FF5733 !important;  /* Оранжевый цвет чисел */
            font-weight: bold;
        }

         /* Отступ между слайдерами */
        div[data-baseweb="slider"] {
            margin-bottom: 30px; /* Увеличиваем расстояние между слайдерами */
        }

        /* Меняем цвет текста у всех лейблов слайдеров */
        div[data-testid="stWidgetLabel"] {{
            color: {random_color} !important; /* Цвет текста */
            font-weight: bold;
            font-size: 16px;
        }}

        /* Стилизация значения слайдера */
        div[data-testid="stSliderValue"] {
            color: #8E44AD !important;  /* Фиолетовый цвет */
            font-size: 18px;
        }
    </style>
""", unsafe_allow_html=True)

max_length = st.slider("Максимальная длина последовательности", 20, 500, value=50)
temperature = st.slider('Температура модели', 0.0, 2.0, value=0.8)
top_k = st.slider('Top-k', 1, 100, value=50)
top_p = st.slider('Top-p', 0.0, 1.0, value=0.9)

st.divider()

if st.button('Сгенерировать весь этот бред!'):
    generated_text = generate_text(prompt, max_length, temperature, top_k, top_p)[0]

    end_time = time.time()
    generation_time = round(end_time - start_time, 2)

    st.markdown(f'**Сгенерированный текст** (это точно текст?):')
    st.markdown(f"""
        <p style='font-family:monospace; color:{random_color}; font-size:24px; font-weight:bold; text-align:center;'>
            <span style='font-size:36px;'>[</span>{generated_text}<span style='font-size:36px;'>]</span>
        </p>
        <div style='text-align:center; border:2px solid #000000; padding:10px; width: fit-content; margin: 10px auto;'>
            <span style='font-size:14px;'>Time taken to generate: {generation_time} seconds</span>
        </div>
""", unsafe_allow_html=True)
