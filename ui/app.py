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

def predict_main_sentiment(tweet, weights, freqs):
    """
    Main prediction function that returns detailed sentiment analysis
    """
    x = extract_features(tweet, freqs)
    y_pred = sigmoid(np.dot(x, weights))
    y_pred_val = y_pred[0][0]
    
    # Determine main sentiment
    if y_pred_val > 0.5:
        main_sentiment = "POSITIVE"
        confidence = y_pred_val * 100
        emoji = "😊"
        color = "green"
    else:
        main_sentiment = "NEGATIVE"  
        confidence = (1 - y_pred_val) * 100
        emoji = "😞"
        color = "red"
    
    return main_sentiment, confidence, emoji, color, y_pred_val

# -----------------------------
# Single Input Prediction
# -----------------------------
user_input = st.text_input("Enter sentence/phrase:")

if user_input:
    # Get main prediction
    main_sentiment, confidence, emoji, color, raw_prob = predict_main_sentiment(user_input, w, freqs)
    
    # Display main prediction prominently
    st.markdown(f"## 🎯 Main Prediction: {emoji} **{main_sentiment}**")
    
    # Create columns for better layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Sentiment", main_sentiment)
    
    with col2:
        st.metric("Confidence", f"{confidence:.1f}%")
    
    with col3:
        st.metric("Raw Score", f"{raw_prob:.3f}")
    
    # Color-coded result
    if main_sentiment == "POSITIVE":
        st.success(f"✅ The text expresses a **POSITIVE** sentiment with {confidence:.1f}% confidence")
    else:
        st.error(f"❌ The text expresses a **NEGATIVE** sentiment with {confidence:.1f}% confidence")
    
    # Additional details
    with st.expander("📊 Detailed Analysis"):
        st.write(f"**Input Text:** {user_input}")
        st.write(f"**Main Sentiment:** {main_sentiment}")
        st.write(f"**Confidence Level:** {confidence:.2f}%")
        st.write(f"**Raw Probability Score:** {raw_prob:.4f}")
        
        # Interpretation guide
        st.write("**Interpretation Guide:**")
        st.write("- Score > 0.5: Positive sentiment")
        st.write("- Score < 0.5: Negative sentiment")
        st.write("- Confidence: How certain the model is about the prediction")
        
        # Processed words preview
        processed_words = process_tweet(user_input)
        if processed_words:
            st.write(f"**Processed Words:** {', '.join(processed_words[:10])}")
            if len(processed_words) > 10:
                st.write(f"... and {len(processed_words) - 10} more words")

# -----------------------------
# File Upload
# -----------------------------
st.markdown("---")
st.markdown("### 📁 Batch Analysis")
uploaded_file = st.file_uploader("Upload a text file with one sentence per line", type=["txt", "csv"])

if uploaded_file is not None:
    lines = uploaded_file.read().decode("utf-8").splitlines()
    lines = [line.strip() for line in lines if line.strip()]  # clean empty lines

    st.write(f"### 📊 Analysis Results for {len(lines)} sentences:")
    
    # Analyze all sentences
    results = []
    positive_count = 0
    negative_count = 0
    
    for line in lines:
        main_sentiment, confidence, emoji, color, raw_prob = predict_main_sentiment(line, w, freqs)
        results.append((line, main_sentiment, confidence, emoji, raw_prob))
        
        if main_sentiment == "POSITIVE":
            positive_count += 1
        else:
            negative_count += 1
    
    # Summary statistics
    st.markdown("#### 📈 Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Sentences", len(lines))
    with col2:
        st.metric("Positive", positive_count, f"{positive_count/len(lines)*100:.1f}%")
    with col3:
        st.metric("Negative", negative_count, f"{negative_count/len(lines)*100:.1f}%")
    with col4:
        overall_sentiment = "POSITIVE" if positive_count > negative_count else "NEGATIVE"
        st.metric("Overall Trend", overall_sentiment)
    
    # Individual results
    st.markdown("#### 📝 Individual Results")
    for i, (line, sentiment, confidence, emoji, raw_prob) in enumerate(results):
        # Create expandable section for each result
        with st.expander(f"{i+1}. {emoji} {sentiment} - {line[:50]}{'...' if len(line) > 50 else ''}"):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**Text:** {line}")
                st.write(f"**Prediction:** {emoji} **{sentiment}**")
            with col2:
                st.write(f"**Confidence:** {confidence:.1f}%")
                st.write(f"**Score:** {raw_prob:.3f}")
    
    # Option to download results
    if st.button("📥 Download Results as CSV"):
        import io
        output = io.StringIO()
        output.write("Text,Sentiment,Confidence,Raw_Score\n")
        for line, sentiment, confidence, emoji, raw_prob in results:
            output.write(f'"{line}",{sentiment},{confidence:.2f},{raw_prob:.4f}\n')
        
        st.download_button(
            label="Download CSV",
            data=output.getvalue(),
            file_name="sentiment_analysis_results.csv",
            mime="text/csv"
        )
