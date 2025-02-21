# ELBRUSE Bootcamp 
# 13-02-2025
# Week 9 Day 4 Project
# team: Dasha, Alina, Ilya, Andrey 

import streamlit as st
import pandas as pd



st.title('Оглавление')

st.write('-------------------------------------------------------------------------------------')

col1, col2, col3 = st.columns(spec=[0.2, 0.6, 0.2])
with col1:
    st.image('images/NLP01.png', width=100)
with col2:
    st.write('Страница 1')    
    st.page_link("pages/page_03.py", label='Классификация отзывов на поликлиники')
with col3:
    st.page_link("pages/page_031.py", label='Описание модели')

st.write('-------------------------------------------------------------------------------------')

col1, col2, col3 = st.columns(spec=[0.2, 0.6, 0.2])
with col1:
    st.image('images/toxic01.png', width=100)
with col2:
    st.write('Страница 2')
    st.page_link("pages/page_04.py", label='Оценка степени токсичности пользовательского сообщения')
with col3:
    st.page_link("pages/page_041.py", label='Описание модели')


st.write('-------------------------------------------------------------------------------------')


col1, col2, col3 = st.columns(spec=[0.2, 0.6, 0.2])
with col1:
    st.image('images/gener01.png', width=100)
with col2:
    st.write('Страница 3')    
    st.page_link("pages/page_05.py", label='Генерация текста GPT-моделью по пользовательскому prompt')
with col3:
    st.page_link("pages/page_051.py", label='Описание модели')

st.write('-------------------------------------------------------------------------------------')
