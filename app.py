import streamlit as st
import pickle
import os

st.title("Fake News Detection 📰")
st.write("Мәтінді енгіз — модель Fake немесе Real деп анықтайды")

# ---- Модель файлының дұрыс жолын анықтау ----
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "model.pkl")

# ---- Модельді жүктеу ----
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model, vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model()

# ---- Интерфейс ----
text = st.text_area("Мәтінді енгіз:", height=150)

if st.button("Тексеру"):
    if not text.strip():
        st.warning("Мәтін енгізіңіз!")
    else:
        X = vectorizer.transform([text])
        pred = model.predict(X)[0]

        if pred == 1:
            st.success("🔵 REAL NEWS")
        else:
            st.error("🔴 FAKE NEWS")
