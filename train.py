import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import os

df = pd.read_csv("data/fake_real_sample.csv")

X = df["text"]
y = df["label"]

vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

os.makedirs("model", exist_ok=True)

with open("model/model.pkl", "wb") as f:
    pickle.dump((model, vectorizer), f)

