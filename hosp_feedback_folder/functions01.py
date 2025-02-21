

# Andrey Abramov
# Elbrus
# 21-02-2025

# external function for model apply


#libs

# common
import pandas as pd
import numpy as np
import random as random

# language processing
import nltk
from nltk.stem.snowball import SnowballStemmer # lemmatization
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer # vectorization
from nltk.corpus import stopwords
nltk.download('punkt_tab')

# mdata preparation
from sklearn.model_selection import train_test_split

# ML models
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
import torch

# timer
import datetime

# metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# saving models
import pickle
from .BERT_function import *



# language processing preparing
nltk.download("stopwords") # stop words removing
nltk.download('punkt') # split the text
nltk.download('wordnet') # lemmatization


def model_loads(model_path_name:str=''):
    answer = 0  
    model = 0  
    try: #[1]
        model = pickle.load(open(model_path_name, 'rb'))
    except:
        pass
    else:
        answer = 1    
    return model, answer

def torch_model_loads(model_path_name:str=''):
    answer = 0  
    model = 0  
    try: 
        model = torch.load(model_path_name)
    except:
        pass
    else:
        answer = 1    
    return model, answer


model_list = [0]*15 # list of loaded models
# load prepared models
model01, model_list[1] = model_loads('hosp_feedback_folder/models/model_01.mod')
model02, model_list[2] = model_loads('hosp_feedback_folder/models/model_02.mod')
model03, model_list[3] = model_loads('hosp_feedback_folder/models/model_03.mod')
model04, model_list[4] = model_loads('hosp_feedback_folder/models/model_04.mod')
model05, model_list[5] = model_loads('hosp_feedback_folder/models/model_05.mod')
model06, model_list[6] = model_loads('hosp_feedback_folder/models/model_06.mod')
model07, model_list[7] = model_loads('hosp_feedback_folder/models/model_07.mod')
model08, model_list[8] = model_loads('hosp_feedback_folder/models/model_08.mod')
model09, model_list[9] = model_loads('hosp_feedback_folder/models/model_09.mod')
model10, model_list[10] = model_loads('hosp_feedback_folder/models/model_10.mod')
model11, model_list[11] = model_loads('hosp_feedback_folder/models/model_11.mod')
model12, model_list[12] = model_loads('hosp_feedback_folder/models/model_12.mod')


model_BERT, model_list[13] = torch_model_loads('hosp_feedback_folder/models/model_bert.mod')


# load special tables
x_BOW_table = pd.read_csv('hosp_feedback_folder/data/x_BOW_names.csv', index_col=0, delimiter=',')
x_len = len(x_BOW_table)
x_line = [0] * x_len
x_line = pd.DataFrame(x_line).T
x_line.columns = x_BOW_table['names']
x_BOW_table = x_line




def models_apply(table:pd.DataFrame, text:str=''):
    start1 = datetime.datetime.now()
    data = pd.DataFrame([text], columns = ['text'])
    data_2_BERT = start1
    
    # clean the content -> only chars =======================================
    for i in range(0, len(data)):
        data.loc[i, 'text'] = re.sub('[^а-яА-Я ]', '', data.loc[i, 'text'])

    # tokenization -> split the text by tokens to list  =======================================
    text_list = []
    for i in range(0, len(data)):
        text_list.append(nltk.word_tokenize(data.loc[i, 'text'], language='russian'))
    
    # lemmatization -> to the base of simple word =======================================
    Snow = SnowballStemmer('russian')

    for i in range(0, len(text_list)):
        for j in range(0, len(text_list[i])):
            text_list[i][j] = Snow.stem(text_list[i][j])

    # stop words removing =======================================
    stop_words_russian = stopwords.words('russian')
    text_list_cleaned = []
    for i in range(0, len(text_list)):
        a = []
        for j in range(0, len(text_list[i])):
            if text_list[i][j] not in stop_words_russian:
                a.append(text_list[i][j])
        text_list_cleaned.append(a)

    # single chars still in the set
    # preparing the set of single chars
    alp='абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
    alp = alp + alp.upper()
    alp  = [i for i in alp]

    text_list_cleaned2 = []
    for i in range(0, len(text_list)):
        a = []
        for j in range(0, len(text_list[i])):
            if text_list[i][j] not in alp:
                a.append(text_list[i][j])
        text_list_cleaned2.append(a)

    # vectorization =======================================
    Vectorizer = CountVectorizer()

    # connect separeted words to the string
    text_list_cleaned3 = []
    for i in range(0, len(text_list)):
        text_list_cleaned3.append(' '.join(text_list_cleaned2[i]))

    # vectorization itself ===============================================
    matrix_count = Vectorizer.fit_transform(text_list_cleaned3)        
    x_BOW_table2 = x_BOW_table
    new_x_BOW = pd.DataFrame(matrix_count.toarray(), columns=[*Vectorizer.get_feature_names_out()])
    x_BOW_table2.loc[0, new_x_BOW.columns] = new_x_BOW.iloc[0,  :]
    x_BOW_table2 = x_BOW_table2.loc[:, x_BOW_table.columns]


    # keep TFI-DF ===============================================
    tfi_vectorizer = TfidfVectorizer()
    tfi_matrix = tfi_vectorizer.fit_transform(raw_documents=text_list_cleaned3)

    x_TFI_table2 = x_BOW_table
    new_x_TFI = pd.DataFrame(tfi_matrix.toarray(), columns=[*tfi_vectorizer.get_feature_names_out()])
    x_TFI_table2.loc[0, new_x_BOW.columns] = new_x_TFI.iloc[0,  :]
    x_TFI_table2 = x_TFI_table2.loc[:, x_BOW_table.columns]



    finish1 = (datetime.datetime.now() - start1).total_seconds()

    table = model_run(model01, table, 1, 0, finish1, x_BOW_table2)
    table = model_run(model02, table, 2, 0, finish1, x_BOW_table2)
    table = model_run(model03, table, 3, 0, finish1, x_BOW_table2)
    table = model_run(model04, table, 4, 0, finish1, x_BOW_table2)
    table = model_run(model05, table, 5, 0, finish1, x_BOW_table2)
    table = model_run(model06, table, 6, 0, finish1, x_BOW_table2)

    table = model_run(model07, table, 1, 1, finish1, x_TFI_table2)
    table = model_run(model08, table, 2, 1, finish1, x_TFI_table2)


    return table


def model_run(model, table, number, set, time_before, x_BOW):
    if model_list[number] ==1 :
        start = datetime.datetime.now()
        table.iloc[number - 1,1 + set] = 'позитив' if model.predict(x_BOW)[0] > 0.5 else 'негатив'
        finish = (datetime.datetime.now() - start).total_seconds()
        if set == 0:
            table.iloc[number - 1,3] = f'{(time_before + finish):.2f}'
    return table




def models_apply2(text):
    #output = BERT_model_apply(model_BERT, text)
    #return model_list[13]
    return ''

