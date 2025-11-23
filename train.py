import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Датасет жүктеу
df = pd.read_csv("data/fake_labeled.csv")  # файл атауын өзіңдікіне ауыстыр

# X және y
X = df["text"]
y = df["label"]

# Text → TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model
with open("model/model.pkl", "wb") as f:
    pickle.dump((model, vectorizer), f)

print("Модель сақталды: model/model.pkl")
