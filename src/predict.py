"""Classify tweets as positive / negative / neutral.

The corpus only has positive and negative labels, so "neutral" is predicted when
the model is not confident: P(positive) falls inside [NEUTRAL_LOW, NEUTRAL_HIGH].
"""
import sys

import joblib

from preprocess import clean_tweet

NEUTRAL_LOW, NEUTRAL_HIGH = 0.40, 0.60


def predict(model, text: str):
    p_pos = model.predict_proba([clean_tweet(text)])[0][1]
    if p_pos >= NEUTRAL_HIGH:
        return "positive", p_pos
    if p_pos <= NEUTRAL_LOW:
        return "negative", 1 - p_pos
    return "neutral", 1 - abs(p_pos - 0.5) * 2


if __name__ == "__main__":
    model = joblib.load("../models/sentiment.joblib")
    tweets = sys.argv[1:] or [
        "I love this new phone, the camera is amazing! :)",
        "Worst service ever, totally disappointed :(",
        "The meeting is scheduled for 3 pm tomorrow.",
    ]
    for t in tweets:
        label, conf = predict(model, t)
        print(f"{label:9s} ({conf:.2f})  {t}")
