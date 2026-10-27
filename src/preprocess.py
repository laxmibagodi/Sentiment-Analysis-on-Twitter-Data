"""Tweet cleaning: URLs, mentions, hashtags, stopwords, stemming."""
import re
import string

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

_tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
_stemmer = PorterStemmer()
# Keep negations: they carry sentiment ("not good").
_STOP = set(stopwords.words("english")) - {"not", "no", "nor", "never"}


def clean_tweet(text: str) -> str:
    text = re.sub(r"https?://\S+|www\.\S+", "", text)  # URLs
    text = re.sub(r"^RT[\s]+", "", text)               # retweet marker
    text = text.replace("#", "")                       # keep hashtag word
    tokens = _tokenizer.tokenize(text)
    out = []
    for tok in tokens:
        if tok in _STOP:
            continue
        if tok in string.punctuation and tok not in {":)", ":("}:
            continue
        out.append(_stemmer.stem(tok))
    return " ".join(out)
