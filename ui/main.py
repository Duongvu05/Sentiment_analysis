import re
import string
import numpy as np
import streamlit as st
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import nltk

# -----------------------------
# NLTK Setup
# -----------------------------
nltk.download("stopwords")

# -----------------------------
# Streamlit App Title
# -----------------------------
st.title("Sentiment Analysis App")
st.write("Enter a single sentence or phrase to analyze its sentiment.")

# -----------------------------
# Load Logistic Regression Assets
# -----------------------------
@st.cache_resource
def load_logreg_assets():
    w = np.load("sentiment_analysis_logistic_weights.npy")  # shape: (3, 1)
    freqs = np.load("vocab.npy", allow_pickle=True).item()
    return w, freqs

w, freqs = load_logreg_assets()

# -----------------------------
# Precompile Regex
# -----------------------------
RE_PATTERNS = {
    "cash": re.compile(r"\$\w+"),
    "rt": re.compile(r"^RT[\s]+"),
    "url": re.compile(r"https?://[^\s\n\r]+"),
    "hash": re.compile(r"#"),
    "elong": re.compile(r"(.)\1{2,}"),
    "exclaim": re.compile(r"!{2,}"),
    "question": re.compile(r"\?{2,}")
}

# -----------------------------
# Initialize Tools
# -----------------------------
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
stemmer = PorterStemmer()
stopwords_english = set(stopwords.words("english"))

# -----------------------------
# Helper Functions
# -----------------------------
def handle_negations(tokens):
    negation_words = {"not", "n't"}
    new_tokens, negate = [], False
    for token in tokens:
        if token in negation_words:
            negate = True
            continue
        if negate:
            new_tokens.append("NOT_" + token)
            negate = False
        else:
            new_tokens.append(token)
    return new_tokens

def process_tweet(tweet: str):
    tweet = RE_PATTERNS["cash"].sub("", tweet)
    tweet = RE_PATTERNS["rt"].sub("", tweet)
    tweet = RE_PATTERNS["url"].sub("", tweet)
    tweet = RE_PATTERNS["hash"].sub("", tweet)
    tweet = RE_PATTERNS["elong"].sub(r"\1\1", tweet)
    tweet = RE_PATTERNS["exclaim"].sub(" multi_exclaim ", tweet)
    tweet = RE_PATTERNS["question"].sub(" multi_question ", tweet)

    tokens = tokenizer.tokenize(tweet)
    tokens = handle_negations(tokens)
    tokens = [word for word in tokens if word not in string.punctuation]

    return [
        stemmer.stem(word)
        for word in tokens
        if (word not in stopwords_english) or word.startswith("NOT_")
    ]

def extract_features(tweet, freqs):
    words = process_tweet(tweet)
    x = np.zeros((1, 3))
    x[0, 0] = 1  # Bias term
    for word in words:
        x[0, 1] += freqs.get((word, 1), 0)  # Positive count
        x[0, 2] += freqs.get((word, 0), 0)  # Negative count
    return x

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict_sentiment(tweet, weights, freqs):
    x = extract_features(tweet, freqs)
    y_pred = sigmoid(np.dot(x, weights))
    y_pred_val = y_pred[0][0]
    sentiment = "Positive" if y_pred_val > 0.5 else "Negative"
    confidence = max(y_pred_val, 1 - y_pred_val) * 100  # as percentage
    return sentiment, confidence

# -----------------------------
# Single Input Prediction
# -----------------------------
user_input = st.text_input("Enter sentence/phrase:")

if user_input:
    sentiment, confidence = predict_sentiment(user_input, w, freqs)
    st.write(f"**Sentiment:** {sentiment}")
    st.write(f"**Confidence Level:** {confidence:.2f}%")

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader("Or upload a text file with one sentence per line", type=["txt", "csv"])

if uploaded_file is not None:
    lines = uploaded_file.read().decode("utf-8").splitlines()
    lines = [line.strip() for line in lines if line.strip()]  # clean empty lines

    st.write("### Sentiment Results:")
    results = [predict_sentiment(line, w, freqs) for line in lines]

    for i, (line, (sentiment, confidence)) in enumerate(zip(lines, results)):
        st.write(f"{i+1}. {line} → **{sentiment}** (Confidence level: {confidence:.2f}%)")
