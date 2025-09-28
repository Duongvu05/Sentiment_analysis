# Sentiment Analysis App

A simple **Sentiment Analysis** web application built with **Streamlit** that predicts the sentiment of sentences or phrases as **Positive** or **Negative** using a logistic regression model. The app also displays the **confidence level** of each prediction.

---

## 🚀 Features

- **Single Sentence Analysis**: Analyze individual sentences or phrases instantly
- **Batch File Processing**: Upload a file with multiple sentences for bulk analysis
- **Custom Logistic Regression Model**: Uses precomputed word frequencies and trained weights
- **Confidence Scoring**: Shows prediction confidence as a percentage
- **Interactive Web Interface**: Simple, user-friendly Streamlit-based UI
- **Real-time Processing**: Instant sentiment predictions with preprocessing pipeline

---

## 📦 Required Packages

Make sure you have the following packages installed:

- `streamlit` - Web application framework
- `numpy` - Numerical computations
- `nltk` - Natural language processing toolkit

### Installation

```bash
pip install streamlit numpy nltk
```

**Download NLTK stopwords** (automatically done in the app, but can be done manually once):

```python
import nltk
nltk.download("stopwords")
```

---

## 📁 Required Files

Make sure these files are in the **same directory** as `app.py`:

- **`sentiment_analysis_logistic_weights.npy`** – Trained logistic regression weights
- **`vocab.npy`** – Precomputed word frequencies for feature extraction
- **`app.py`** – Main Streamlit application file

### File Structure
```
ui/
├── README.md                              # This documentation
├── app.py                                # Main Streamlit application
├── sentiment_analysis_logistic_weights.npy # Model weights
└── vocab.npy                             # Vocabulary frequencies
```

---

## 🖥️ How to Run Locally

### Method 1: Using Command Line / Terminal

1. **Open a terminal or command prompt**

2. **Navigate to your project folder:**
   ```bash
   cd path/to/your/project/ui/folder
   ```

3. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

4. **Access the app:**
   - The app will open automatically in your default browser at `http://localhost:8501`
   - If it doesn't open automatically, manually navigate to the URL shown in terminal

### Method 2: Using VS Code / PyCharm

1. **Open your project** in VS Code or PyCharm
2. **Open the built-in terminal**
3. **Navigate to the ui directory:**
   ```bash
   cd ui
   ```
4. **Run the application:**
   ```bash
   streamlit run app.py
   ```

The app will open in your browser as described above.

---

## 🎯 How to Use the App

### Single Sentence Analysis
1. Enter a sentence or phrase in the **text input box**
2. Click **"Analyze Sentiment"** or press Enter
3. View the predicted **sentiment** (Positive/Negative) and **confidence level**

### Batch File Analysis
1. Click **"Upload a file"** button
2. Select a `.txt` or `.csv` file containing sentences (one per line)
3. View sentiment predictions for all sentences with confidence scores

### Example Usage
- **Input**: "I love this movie!"
- **Output**: Positive sentiment with 95.2% confidence

- **Input**: "This is terrible and disappointing."
- **Output**: Negative sentiment with 87.8% confidence

---

## 🔧 Technical Implementation

### Model Architecture
- **Algorithm**: Custom Logistic Regression
- **Features**: Frequency-based word counts (positive/negative frequencies)
- **Preprocessing**: Advanced text cleaning and normalization
- **Weights**: Pre-trained model weights stored in NumPy format

### Text Preprocessing Pipeline
The app includes comprehensive text preprocessing:

- **URL Removal**: Removes web links and URLs
- **Hashtag Processing**: Handles Twitter hashtags and mentions
- **Cash Mentions**: Processes financial symbols and cash mentions
- **Negation Handling**: Properly processes negative constructions
- **Word Stemming**: Reduces words to their root forms
- **Stopword Removal**: Removes common words (the, and, or, etc.)
- **Normalization**: 
  - Handles elongated words (e.g., "sooooo" → "so")
  - Normalizes repeated punctuation (e.g., "!!!" → "!")
  - Converts to lowercase

### Confidence Calculation
The confidence level is derived from the logistic regression probability:
- Values closer to 0% or 100% indicate higher confidence
- Values around 50% indicate uncertainty in the prediction

---

## 📊 Performance Characteristics

- **Accuracy**: Based on the trained logistic regression model (typically 95%+ on test data)
- **Speed**: Real-time prediction for single sentences
- **Scalability**: Efficient batch processing for multiple sentences
- **Memory Usage**: Lightweight with pre-loaded model weights

---

## 🚨 Troubleshooting

### Common Issues

1. **Missing Files Error**
   ```
   FileNotFoundError: No such file or directory: 'vocab.npy'
   ```
   **Solution**: Ensure `vocab.npy` and `sentiment_analysis_logistic_weights.npy` are in the same directory as `app.py`

2. **NLTK Data Error**
   ```
   LookupError: Resource stopwords not found
   ```
   **Solution**: Run `nltk.download("stopwords")` in Python or let the app download it automatically

3. **Port Already in Use**
   ```
   OSError: [Errno 48] Address already in use
   ```
   **Solution**: Use a different port with `streamlit run app.py --server.port 8502`

4. **Package Import Error**
   ```
   ModuleNotFoundError: No module named 'streamlit'
   ```
   **Solution**: Install required packages with `pip install streamlit numpy nltk`

---

## 🎨 Customization

### Modifying the UI
- Edit `app.py` to customize the Streamlit interface
- Add new input methods or visualization components
- Modify styling with Streamlit's theming options

### Updating the Model
- Replace `sentiment_analysis_logistic_weights.npy` with new trained weights
- Update `vocab.npy` with new vocabulary frequencies
- Ensure compatibility with the existing preprocessing pipeline

---

## 📈 Future Enhancements

Potential improvements for the application:

- **Multiple Model Support**: Add support for different ML models
- **Advanced Visualizations**: Sentiment trends and confidence distributions  
- **Export Functionality**: Save results to CSV or PDF
- **API Integration**: REST API endpoints for programmatic access
- **Real-time Twitter Analysis**: Connect to social media streams
- **Multi-language Support**: Extend to other languages

---

## 📄 License

This project is licensed under the MIT License. See the main project LICENSE file for details.

---

## 🤝 Contributing

Feel free to submit issues and enhancement requests! This UI is part of the larger Sentiment Analysis project.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes to the UI components
4. Test the Streamlit application thoroughly
5. Submit a pull request

---

## 📞 Support

For questions or issues:
- Check the troubleshooting section above
- Review the main project documentation
- Submit an issue on the GitHub repository

---

*This Streamlit application provides an intuitive interface for the comprehensive sentiment analysis project, making machine learning predictions accessible through a simple web interface.*
