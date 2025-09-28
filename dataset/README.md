# Dataset Information

This directory contains datasets for the sentiment analysis project.

## Structure:
- `raw/` - Original, unprocessed datasets
- `processed/` - Cleaned and preprocessed datasets ready for training

## Datasets Description:

### Raw Data
- Place your original datasets (.csv, .json, .txt) in the `raw/` directory
- Common sentiment datasets:
  - IMDB Movie Reviews
  - Amazon Product Reviews
  - Twitter Sentiment Data
  - Custom collected data

### Processed Data
- Cleaned and preprocessed versions of raw data
- Features may include:
  - Text normalization
  - Tokenization
  - Encoding labels
  - Train/validation/test splits

## Data Format
Expected format for processed data:
- `train.csv` - Training set
- `val.csv` - Validation set  
- `test.csv` - Test set

Each CSV should contain at least:
- `text` column: The text to analyze
- `label` column: Sentiment label (0=negative, 1=positive, etc.)