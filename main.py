# ELBRUSE Bootcamp 
# 13-02-2025
# Week 9 Day 4 Project
# team: Dasha, Alina, Ilya, Andrey u

import streamlit as st
import pandas as pd



#initialization ----------------------------


#Основная страница  ----------------------------
# боковая панель
page01 = st.Page("pages/page_01.py", title = 'Оглавление ->')

page02 = st.Page("pages/page_02.py", title = 'Описание Проекта')
page03 = st.Page("pages/page_03.py", title = '1. Классификация отзывов на поликлиники')
page04 = st.Page("pages/page_031.py", title = ' - описание модели')
page05 = st.Page("pages/page_04.py", title = '2. Оценка степени токсичности пользовательского сообщения')
page06 = st.Page("pages/page_041.py", title = ' - описание модели')
page07 = st.Page("pages/page_05.py", title = '3. Генерация текста GPT-моделью по пользовательскому prompt')
page08 = st.Page("pages/page_051.py", title = ' - описание модели')


pg = st.navigation([page01,  page02, 
                    page03, page04,
                    page05, page06,
                    page07, page08
                    ], expanded=True)
pg.run()


st.sidebar.title('Команда проекта:')
st.sidebar.write('Алина Зарницина')
st.sidebar.write('Нанзат Дашиев')
st.sidebar.write('Андрей Абрамов')

    


    






