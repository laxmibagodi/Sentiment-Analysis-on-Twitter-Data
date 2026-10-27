"""Train on Sentiment140 (Kaggle). Run from src/:  python train_kaggle.py"""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

nltk.download("stopwords", quiet=True)
from preprocess import clean_tweet

PATH = "../data/training.1600000.processed.noemoticon.csv"
SAMPLE = 200_000  # full 1.6M works too but cleaning takes a long time

df = pd.read_csv(PATH, encoding="latin-1", header=None,
                 names=["target", "id", "date", "flag", "user", "text"])
df = df[["target", "text"]].sample(SAMPLE, random_state=42)
df["label"] = (df["target"] == 4).astype(int)   # 0 = negative, 4 = positive
print(df["label"].value_counts())

df["clean"] = df["text"].map(clean_tweet)
X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])

model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)),
    ("clf", LogisticRegression(max_iter=1000, C=2.0)),
])
model.fit(X_train, y_train)

pred = model.predict(X_test)
print(classification_report(y_test, pred, target_names=["negative", "positive"]))
ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["negative", "positive"], cmap="Blues")
plt.savefig("../confusion_matrix.png", dpi=150, bbox_inches="tight")
joblib.dump(model, "../models/sentiment.joblib")