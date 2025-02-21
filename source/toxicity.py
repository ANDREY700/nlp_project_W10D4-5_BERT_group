import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch import nn
import streamlit as st

PATH = "./../models/toxicity.pt"

@st.cache_data
def load_model(path):
    model_checkpoint = 'cointegrated/rubert-tiny-toxicity'
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)

    model.classifier = nn.Sequential(
        nn.Linear(in_features=312, out_features=64, bias=True),
        nn.Linear(in_features=64, out_features=1, bias=True)
    )
    model.load_state_dict(torch.load(path, weights_only=True, map_location=torch.device('cpu')))
    model.eval()
    return model, tokenizer

def text2toxicity(text, model, tokenizer):
    """ Calculate toxicity """

    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(model.device)
        proba = torch.sigmoid(model(**inputs).logits).cpu().numpy()
    return proba.squeeze(-1)

#
# print(text2toxicity(['я люблю украинцев', 'я ненавижу хохлов'], PATH))
# # 0.9350118728093193