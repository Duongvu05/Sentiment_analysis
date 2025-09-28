# Sentiment Analysis App

A simple **Sentiment Analysis** web application built with **Streamlit** that predicts the sentiment of sentences or phrases as **Positive** or **Negative** using a logistic regression model. The app also displays the **confidence level** of each prediction.

---

## Features

- Analyze single sentences or upload a file with multiple sentences.
- Uses a custom logistic regression model with precomputed word frequencies.
- Shows confidence level as a percentage for each prediction.
- Simple, interactive, and easy-to-use web interface.

---

## Required Packages

- `streamlit`  
- `numpy`  
- `nltk`  

**Download NLTK stopwords** (automatically done in the app, but can be done manually once):

```python
import nltk
nltk.download("stopwords")
```

## Files Needed

- `sentiment_analysis_logistic_weights.npy` – Logistic regression weights.  
- `vocab.npy` – Precomputed word frequencies for features.  
- `app.py` – Main Streamlit application file.  

Make sure these files are in the **same directory** as `app.py`.


## How to Run Locally

### 1. Using Command Line / Terminal

1. Open a terminal or command prompt.  
2. Navigate to your project folder:

```bash
cd path/to/your/project/folder
```
 
3.Run the Streamlit app:

```bash
streamlit run app.py
```

4. The app will open automatically in your default browser at http://localhost:8501.

**After running the app, you can:**

- Enter a sentence in the input box to see the predicted **sentiment** and **confidence level**.  
- Upload a `.txt` or `.csv` file containing multiple sentences (one per line) to get batch sentiment predictions.


### 2. Using VS Code / PyCharm

Open your project in VS Code or PyCharm.

Open the built-in terminal.

Run:

```bash
streamlit run app.py
```

The app will open in your browser as above.

## Notes

- The confidence level is derived from the logistic regression probability of the predicted class.  

### Preprocessing includes:

- Removing URLs, hashtags, cash mentions.  
- Handling negations.  
- Stemming words.  
- Removing stopwords.  
- Normalizing elongated words and repeated punctuation.

---

## License

This project is licensed under the MIT License.


