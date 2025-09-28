# Sentiment Analysis - Comprehensive Homework Solutions

This repository contains a complete implementation of sentiment analysis algorithms and comprehensive homework solutions covering multiple machine learning approaches for text classification.

## 📊 Project Overview

This project demonstrates a thorough understanding of sentiment analysis through:
- **Custom logistic regression implementation** from scratch
- **Comparative analysis** with scikit-learn implementations  
- **Feature engineering** and normalization techniques
- **Numerical stability** solutions for gradient descent
- **Advanced ML model comparisons** with 17 different algorithms
- **Data preprocessing** and scaling techniques

## 📂 Project Structure 
```
Sentiment_analysis/
├── sentiment_analysis_homework_solutions.ipynb  # Complete homework solutions
├── pyproject.toml                              # Project configuration and dependencies
├── uv.lock                                     # Lock file for dependencies
├── README.md                                   # This documentation
├── config/                                     # Configuration files for training
├── dataset/                                    # Data directory
│   ├── raw/                                   # Raw datasets
│   ├── processed/                             # Processed datasets
│   └── README.md                              # Dataset descriptions
└── src/                                       # Source code modules
    ├── __init__.py
    ├── data/                                  # Data processing utilities
    ├── models/                                # Model definitions
    ├── training/                              # Training logic
    ├── evaluation/                            # Model evaluation
    └── utils/                                 # Common utilities
```

## 🎯 Assignment Solutions Overview

### **Exercise 1: Custom vs Scikit-learn Logistic Regression**
- ✅ Implemented custom logistic regression from scratch
- ✅ Comprehensive comparison with sklearn implementations
- ✅ Performance analysis across multiple configurations
- **Result**: Custom implementation achieved **99.60%** accuracy

### **Exercise 2: Gradient Descent Numerical Stability**  
- ✅ Identified and analyzed overflow issues in high-iteration training
- ✅ Implemented numerically stable sigmoid and cost functions
- ✅ Added gradient clipping and early stopping mechanisms
- **Result**: Successfully trained with 100K iterations without errors

### **Exercise 3: Feature Normalization with Sentence Length**
- ✅ Implemented normalization by N = train_set_length × sentence_length
- ✅ Statistical comparison of feature distributions
- ✅ Impact analysis on model performance
- **Result**: Original features performed better (**99.50%** vs **96.40%**)

### **Exercise 4: Feature Scaling Techniques**
- ✅ Tested MinMax, Standard, and Robust scaling methods
- ✅ Visual comparison of feature distributions
- ✅ Performance impact analysis
- **Result**: Standard scaling achieved best performance (**99.55%**)

### **Exercise 5: Alternative Decision Functions**
- ✅ Implemented simple frequency-based classifier
- ✅ Detailed comparison with logistic regression
- ✅ Disagreement analysis between methods
- **Result**: Simple rule achieved **99.65%** accuracy

### **Exercise 7: Comprehensive ML Model Comparison**
- ✅ Tested 17 different ML algorithms including:
  - Logistic Regression variants
  - Random Forest and Decision Trees
  - SVM with different kernels
  - Gradient Boosting and AdaBoost
  - K-Nearest Neighbors
  - Naive Bayes variants
  - Neural Networks (MLP)
- ✅ Performance metrics analysis (Accuracy, Precision, Recall, F1-Score)
- ✅ Model complexity vs performance visualization
- **Result**: Ridge Classifier achieved best accuracy (**99.61%**)

## 📈 Key Performance Results

| Method | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|---------|----------|
| **Custom Logistic Regression** | **99.60%** | 99.20% | 99.80% | 99.50% |
| Sklearn Logistic Regression | 99.50% | 99.20% | 99.80% | 99.50% |
| Simple Frequency Classifier | 99.65% | N/A | N/A | N/A |
| **Best ML Model (Ridge)** | **99.61%** | 98.84% | 100% | 99.41% |
| Random Forest (Tuned) | 99.50% | 99.40% | 99.50% | 99.44% |
| Neural Network (Large) | 99.00% | 98.00% | 100% | 99.00% |

## 🔧 Technical Implementation Highlights

### **Custom Logistic Regression Features:**
- Sigmoid activation with numerical stability
- Gradient descent optimization 
- Cost function monitoring
- Custom training and prediction methods

### **Numerical Stability Solutions:**
- Epsilon clipping for log computations
- Z-value clipping for sigmoid overflow prevention
- Gradient clipping for training stability
- Early stopping mechanisms

### **Feature Engineering:**
- Frequency-based feature extraction
- Sentence length normalization
- Multiple scaling techniques
- Statistical analysis and visualization

### **Comprehensive Analysis:**
- Model comparison matrices
- Performance visualization charts
- Training time analysis
- Complexity vs performance trade-offs

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Notebook
- Required packages: numpy, pandas, scikit-learn, matplotlib, seaborn, nltk

### Installation
```bash
# Clone the repository
git clone https://github.com/Duongvu05/Sentiment_analysis.git
cd Sentiment_analysis

# Install dependencies using uv (recommended)
uv sync

# Or install with pip
pip install -e .
```

### Running the Homework Solutions
```bash
# Open the comprehensive solutions notebook
jupyter notebook sentiment_analysis_homework_solutions.ipynb
```

## 📊 Visualizations and Analysis

The notebook includes comprehensive visualizations:
- **Performance comparison charts** across all methods
- **Feature distribution histograms** for different scaling techniques
- **Model complexity analysis** plots
- **Training time comparisons**
- **Confusion matrices** and error analysis
- **Weight comparison visualizations**

## 🎯 Key Learnings and Insights

1. **Custom implementations can compete with library solutions** when properly optimized
2. **Numerical stability is crucial** for high-iteration training scenarios
3. **Feature scaling significantly impacts** certain algorithms more than others
4. **Simple rule-based approaches** can be surprisingly effective for well-structured problems
5. **Advanced ensemble methods** don't always outperform simpler approaches on clean datasets
6. **Proper evaluation and comparison methodology** is essential for reliable results

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---
*This project demonstrates comprehensive understanding of machine learning fundamentals, from custom algorithm implementation to advanced model comparison and analysis.*
