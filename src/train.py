"""Train a TF-IDF + Logistic Regression sentiment classifier on NLTK twitter_samples."""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nltk  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, classification_report  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from preprocess import clean_tweet  # noqa: E402

for pkg in ("twitter_samples", "stopwords"):
    nltk.download(pkg, quiet=True)
from nltk.corpus import twitter_samples  # noqa: E402

pos = twitter_samples.strings("positive_tweets.json")
neg = twitter_samples.strings("negative_tweets.json")
texts = [clean_tweet(t) for t in pos + neg]
labels = [1] * len(pos) + [0] * len(neg)

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
    ("clf", LogisticRegression(max_iter=1000, C=2.0)),
])
model.fit(X_train, y_train)

pred = model.predict(X_test)
print(classification_report(y_test, pred, target_names=["negative", "positive"]))

ConfusionMatrixDisplay.from_predictions(
    y_test, pred, display_labels=["negative", "positive"], cmap="Blues"
)
plt.title("Sentiment classifier - confusion matrix")
plt.savefig("../confusion_matrix.png", dpi=150, bbox_inches="tight")

joblib.dump(model, "../models/sentiment.joblib")
print("Saved models/sentiment.joblib")
