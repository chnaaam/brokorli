import json
import requests
import streamlit as st
	
API_SERVER_URL = "http://localhost:5001" 

st.title('KORIA Framework Demo')

st.header("1. Named Entity Recognition")
ner_sentence = st.text_input("Sentence", key="ner-sentence")
ner_btn_submit = st.button("Submit", key="ner-submit")
if ner_btn_submit:
    response = requests.get(API_SERVER_URL + "/ner", params={"sentence": ner_sentence})
    
    st.json(response.json())


st.header("2. Machine Reading Comprehension")
mrc_sentence = st.text_input("Sentence", key="mrc-sentence")
mrc_question = st.text_input("Question", key="mrc-question")
mrc_btn_submit = st.button("Submit", key="mrc-submit")
if mrc_btn_submit:
    response = requests.get(API_SERVER_URL + "/mrc", params={"sentence": mrc_sentence, "question": mrc_question})
    
    st.json(response.json())