# streamlit_app.py
import streamlit as st
import requests
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

st.title("IMDB Sentiment Analysis")

text = st.text_area("Enter review")

if st.button("Predict"):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=200)
    payload = {"input": padded[0].tolist()}

    res = requests.post("http://localhost:8000/predict", json=payload)
    st.json(res.json())