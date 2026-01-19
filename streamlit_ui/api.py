import streamlit as st
import requests

st.title("Sentiment Analysis (Rust + ONNX)")

text = st.text_area("Some text")

def dummy_tokenize(text):
    return [1]*100

if st.button(""):
    tokens = dummy_tokenize(text)
    res = requests.post("http://localhost:8000/predict", json={"input": tokens})
    st.write("1:", res.json()["prediction"])