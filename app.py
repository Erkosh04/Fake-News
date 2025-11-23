import streamlit as st
import pickle

# Модельді жүктеу
with open("model/model.pkl", "rb") as f:
    model, vectorizer = pickle.load(f)

st.title("Fake News Detection 📰")
st.write("Мәтінді енгіз — модель Fake немесе Real деп анықтайды")

text = st.text_area("Мәтінді енгіз:")

if st.button("Тексеру"):
    if text.strip() == "":
        st.warning("Мәтін жаз!")
    else:
        X = vectorizer.transform([text])
        pred = model.predict(X)[0]
        
        if pred == 1:
            st.success("🔵 REAL NEWS")
        else:
            st.error("🔴 FAKE NEWS")
