# 📊 Dataset Information

This directory contains the datasets used for the comprehensive sentiment analysis homework solutions project.

## 📂 Directory Structure
```
dataset/
├── README.md              # This documentation
├── raw/                   # Original, unprocessed datasets
│   ├── twitter_samples.json   # Twitter sentiment data (JSON format)
│   └── twitter_samples.pkl    # Twitter sentiment data (Pickle format)
└── processed/             # Cleaned and preprocessed datasets (generated during runtime)
```

## 🐦 Twitter Samples Dataset

### Overview
The primary dataset used in this project is the **Twitter Samples Dataset**, which contains labeled tweets for sentiment analysis training and testing.

### Dataset Characteristics
- **Source**: Twitter social media platform
- **Task**: Binary sentiment classification
- **Labels**: 
  - `0` = Negative sentiment
  - `1` = Positive sentiment
- **Format**: Available in both JSON and Pickle formats
- **Size**: Suitable for educational and research purposes

### File Descriptions

#### `twitter_samples.json`
- **Format**: JSON (JavaScript Object Notation)
- **Structure**: List of tweet objects with text and labels
- **Usage**: Human-readable format, easy to inspect
- **Encoding**: UTF-8 for emoji and special character support

#### `twitter_samples.pkl`
- **Format**: Python Pickle (binary serialization)
- **Structure**: Optimized Python objects
- **Usage**: Faster loading for Python applications
- **Advantages**: Preserves exact Python data types and structures

## 🛠️ Data Processing Pipeline

The homework solutions notebook processes the raw data through several stages:

### 1. **Data Loading**
```python
# Example loading from JSON
with open('dataset/raw/twitter_samples.json', 'r') as f:
    data = json.load(f)

# Example loading from Pickle
with open('dataset/raw/twitter_samples.pkl', 'rb') as f:
    data = pickle.load(f)
```

### 2. **Text Preprocessing**
- **Tokenization**: Split text into individual words
- **Lowercasing**: Convert all text to lowercase
- **Stop word removal**: Remove common words (the, and, or, etc.)
- **Punctuation handling**: Remove or normalize punctuation
- **URL/mention removal**: Clean social media specific content

### 3. **Feature Extraction**
- **Frequency-based features**: Count positive/negative word frequencies
- **Bag-of-words representation**: Convert text to numerical features
- **Custom feature engineering**: Sentence length, normalization factors

### 4. **Data Splitting**
- **Training set**: Used to train models (typically 80%)
- **Test set**: Used for final evaluation (typically 20%)
- **Stratified splitting**: Maintains label distribution across splits

## 📈 Dataset Statistics

The processed dataset provides:
- **Balanced classes**: Roughly equal positive and negative samples
- **Rich vocabulary**: Diverse Twitter language patterns
- **Real-world challenges**: Informal language, abbreviations, emojis
- **Educational value**: Perfect size for learning and experimentation

## 🔄 Runtime Data Generation

The `processed/` directory is populated during notebook execution with:
- **Feature matrices**: Numerical representations of text data
- **Frequency dictionaries**: Word-label frequency mappings
- **Preprocessed text**: Cleaned and tokenized versions
- **Train/test splits**: Organized data partitions

## 💡 Usage in Homework Solutions

This dataset is used across all exercises:
1. **Custom logistic regression** training and evaluation
2. **Scikit-learn model** comparisons
3. **Feature engineering** experiments
4. **Scaling technique** analysis
5. **Alternative classifier** development
6. **Comprehensive ML** model benchmarking

## 🎯 Expected Data Format

For compatibility with the homework solutions, data should follow this structure:
```python
# Tweet structure
{
    "text": "I love this movie! It's amazing!",
    "label": 1  # 1 for positive, 0 for negative
}
```

## 📋 Data Quality Notes

- **Preprocessing handled**: All text cleaning is performed in the notebook
- **Label validation**: Binary classification (0/1) is expected
- **Encoding support**: Full Unicode support for international content
- **Memory efficient**: Optimized for educational hardware constraints

---
*This dataset enables comprehensive sentiment analysis learning from basic implementations to advanced ML model comparisons.*