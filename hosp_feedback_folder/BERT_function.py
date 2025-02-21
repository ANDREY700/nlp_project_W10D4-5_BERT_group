



# Andrey Abramov
# Elbrus
# 21-02-2025

# external function for BERT model apply


import pandas as pd
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import torch
from torch import nn
import matplotlib.pyplot as plt

# импортируем трансформеры
import transformers
from transformers import AdamW
import warnings
warnings.filterwarnings('ignore')
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import confusion_matrix


from torch.utils.data import DataLoader, TensorDataset, DataLoader, RandomSampler
from sklearn.utils.class_weight import compute_class_weight
from torchmetrics.classification import BinaryAccuracy

#some useful lybs
import random as random

# metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# подгружаем токенизатор и модель
tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
local_bert = AutoModel.from_pretrained("cointegrated/rubert-tiny2")


class BERT_architecture(nn.Module):
    def __init__(self, bert):     
      super(BERT_architecture, self).__init__()
      self.bert = bert 
      # dropout layer
      self.dropout = nn.Dropout()      
      self.tahn = nn.Tanh()
      self.Sigmoid1 = nn.Sigmoid()
      self.Sigmoid2 = nn.Sigmoid()
      self.fc1 = nn.Linear(312,256)
      self.fc2 = nn.Linear(256,100)
      self.fc3 = nn.Linear(100,32)
      self.fc4 = nn.Linear(32,1)
      #self.fc3 = nn.Linear(64,5)

      self.softmax = nn.LogSoftmax(dim=1)
      
    #define the forward pass
    def forward(self, sent_id, mask):
      #pass the inputs to the model  
      _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False) 
      cls_hs = nn.functional.normalize(cls_hs)
      #print(f'cls : {cls_hs.shape}')  #cls : torch.Size([64, 312])
      x = self.fc1(cls_hs)
      x = self.Sigmoid1(x)
      x = self.dropout(x)

      x = self.fc2(x)
      x = self.Sigmoid2(x)

      x = self.fc3(x)
      x = self.Sigmoid2(x)
      x = self.fc4(x)
      x = self.Sigmoid2(x)
      return x



model_BERT = BERT_architecture(local_bert)



#def models_apply(table:pd.DataFrame, text:str='', model):
def BERT_model_apply(model_dict, text:str=''):
    
    
    model_BERT.load_state_dict(torch.load('/home/andrey/Documents/nlp_project_W10D4-5_BERT_group/hosp_feedback_folder/models/model_bert.mod'))
    model_BERT.eval()

    data = pd.DataFrame([text], columns = ['content'])

    SEQ_LEN = 550 #2048 max
    BATCH_SIZE = 256

    tokens_train_df = tokenizer.batch_encode_plus([text], max_length = SEQ_LEN, pad_to_max_length=True, truncation=True)
    train_X_tenzor = torch.tensor(tokens_train_df['input_ids'])
    train_mask_tenzor = torch.tensor(tokens_train_df['attention_mask'])


    test_data = TensorDataset(train_X_tenzor, train_mask_tenzor, train_mask_tenzor)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)
    
    batch_valid = next(iter(test_dataloader))
    inputs, mask, labels = batch_valid 

    with torch.no_grad():
        output = model_BERT(inputs, mask).squeeze()
    

    return output