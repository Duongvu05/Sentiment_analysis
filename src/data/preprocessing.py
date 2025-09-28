"""
Text preprocessing utilities for sentiment analysis

This module contains functions for cleaning and preprocessing text data
including tokenization, stemming, stopword removal, and other text normalization tasks.
"""

import re
import string
import numpy as np
import pandas as pd
from typing import List, Union, Dict, Any
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import nltk

# Download required NLTK data
def download_nltk_dependencies():
    """Download required NLTK datasets for preprocessing"""
    nltk.download('stopwords', quiet=True)

class TextPreprocessor:
    """Class for text preprocessing operations"""
    
    def __init__(self):
        """Initialize preprocessor with required components"""
        download_nltk_dependencies()
        self.stemmer = PorterStemmer()
        self.tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
        self.stopwords_english = set(stopwords.words('english'))
    
    def process_tweet(self, tweet: str) -> List[str]:
        """Process a single tweet
        
        Args:
            tweet: Raw tweet string
            
        Returns:
            List of processed words
        """
        # Remove stock market tickers like $GE
        tweet = re.sub(r'\$\w*', '', tweet)
        
        # Remove old style retweet text "RT"
        tweet = re.sub(r'^RT[\s]+', '', tweet)
        
        # Remove hyperlinks
        tweet = re.sub(r'https?://[^\s\n\r]+', '', tweet)
        
        # Remove hashtags (only removing the hash # sign from the word)
        tweet = re.sub(r'#', '', tweet)
        
        # Tokenize tweets
        tweet_tokens = self.tokenizer.tokenize(tweet)
        
        tweets_clean = []
        for word in tweet_tokens:
            if (word not in self.stopwords_english and   # remove stopwords
                    word not in string.punctuation):     # remove punctuation
                stem_word = self.stemmer.stem(word)
                tweets_clean.append(stem_word)
        
        return tweets_clean
    
    def process_tweets_batch(self, tweets: List[str]) -> List[List[str]]:
        """Process a batch of tweets
        
        Args:
            tweets: List of raw tweet strings
            
        Returns:
            List of lists of processed words
        """
        return [self.process_tweet(tweet) for tweet in tweets]
    
    def clean_text(self, text: str, remove_punctuation: bool = True, 
                   remove_numbers: bool = False, lowercase: bool = True) -> str:
        """Clean text with various options
        
        Args:
            text: Input text
            remove_punctuation: Whether to remove punctuation
            remove_numbers: Whether to remove numbers
            lowercase: Whether to convert to lowercase
            
        Returns:
            Cleaned text string
        """
        if lowercase:
            text = text.lower()
            
        if remove_numbers:
            text = re.sub(r'\d+', '', text)
            
        if remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
            
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text

class DataPreprocessor:
    """Class for preprocessing dataset splits and saving processed data"""
    
    def __init__(self, raw_data_path: str = "dataset/raw", 
                 processed_data_path: str = "dataset/processed"):
        """Initialize data preprocessor
        
        Args:
            raw_data_path: Path to raw data directory
            processed_data_path: Path to save processed data
        """
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        self.text_preprocessor = TextPreprocessor()
        
        import os
        os.makedirs(processed_data_path, exist_ok=True)
    
    def preprocess_twitter_data(self, train_x: List[str], test_x: List[str],
                               train_y: np.ndarray, test_y: np.ndarray,
                               save_processed: bool = True) -> Dict[str, Any]:
        """Preprocess Twitter dataset
        
        Args:
            train_x: Training texts
            test_x: Test texts
            train_y: Training labels
            test_y: Test labels
            save_processed: Whether to save processed data
            
        Returns:
            Dictionary with processed data
        """
        print("Preprocessing Twitter data...")
        
        # Process tweets
        train_x_processed = self.text_preprocessor.process_tweets_batch(train_x)
        test_x_processed = self.text_preprocessor.process_tweets_batch(test_x)
        
        processed_data = {
            'train': {
                'texts': train_x,
                'processed_texts': train_x_processed,
                'labels': train_y
            },
            'test': {
                'texts': test_x,
                'processed_texts': test_x_processed,
                'labels': test_y
            }
        }
        
        if save_processed:
            self._save_processed_data(processed_data, 'twitter_processed')
            
        print(f"Processed {len(train_x)} training and {len(test_x)} test tweets")
        return processed_data
    
    def _save_processed_data(self, data: Dict[str, Any], filename: str):
        """Save processed data to files"""
        import json
        import pickle
        import os
        
        # Save as JSON
        json_data = {}
        for split, split_data in data.items():
            json_data[split] = {
                'texts': split_data['texts'],
                'processed_texts': split_data['processed_texts'],
                'labels': split_data['labels'].tolist() if isinstance(split_data['labels'], np.ndarray) else split_data['labels']
            }
        
        json_path = os.path.join(self.processed_data_path, f'{filename}.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        # Save as pickle
        pkl_path = os.path.join(self.processed_data_path, f'{filename}.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(data, f)
        
        # Save as CSV files
        for split, split_data in data.items():
            df_data = {
                'text': split_data['texts'],
                'processed_text': [' '.join(tokens) for tokens in split_data['processed_texts']],
                'label': split_data['labels'].flatten() if isinstance(split_data['labels'], np.ndarray) else split_data['labels']
            }
            
            df = pd.DataFrame(df_data)
            csv_path = os.path.join(self.processed_data_path, f'{filename}_{split}.csv')
            df.to_csv(csv_path, index=False)
        
        print(f"Processed data saved to {self.processed_data_path}/")
    
    def load_processed_data(self, filename: str = 'twitter_processed') -> Dict[str, Any]:
        """Load previously processed data
        
        Args:
            filename: Name of the processed data file
            
        Returns:
            Dictionary with processed data
        """
        import pickle
        import os
        
        pkl_path = os.path.join(self.processed_data_path, f'{filename}.pkl')
        
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                return pickle.load(f)
        else:
            raise FileNotFoundError(f"Processed data not found at {pkl_path}")

# Convenience functions
def process_tweet(tweet: str) -> List[str]:
    """Convenience function to process a single tweet"""
    preprocessor = TextPreprocessor()
    return preprocessor.process_tweet(tweet)

def preprocess_twitter_dataset(train_x: List[str], test_x: List[str],
                              train_y: np.ndarray, test_y: np.ndarray) -> Dict[str, Any]:
    """Convenience function to preprocess Twitter dataset"""
    preprocessor = DataPreprocessor()
    return preprocessor.preprocess_twitter_data(train_x, test_x, train_y, test_y)