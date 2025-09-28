"""
Data loading utilities for sentiment analysis project

This module handles loading raw data from various sources including:
- NLTK Twitter samples
- Hugging Face datasets
- Custom data files

The loaded data is saved to dataset/raw/ directory for further processing.
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from typing import List, Tuple, Union
from datasets import load_dataset
from nltk.corpus import twitter_samples
import nltk

# Download required NLTK data
def download_nltk_data():
    """Download required NLTK datasets"""
    nltk.download('twitter_samples', quiet=True)
    nltk.download('stopwords', quiet=True)

class DataLoader:
    """Class for loading and saving raw datasets"""
    
    def __init__(self, raw_data_path: str = "dataset/raw"):
        """Initialize DataLoader
        
        Args:
            raw_data_path: Path to save raw data files
        """
        self.raw_data_path = raw_data_path
        os.makedirs(raw_data_path, exist_ok=True)
        download_nltk_data()
    
    def load_twitter_samples(self, save_to_file: bool = True) -> Tuple[List[str], List[str], np.ndarray, np.ndarray]:
        """Load Twitter samples from NLTK
        
        Args:
            save_to_file: Whether to save the loaded data to files
            
        Returns:
            Tuple of (train_x, test_x, train_y, test_y)
        """
        print("Loading Twitter samples from NLTK...")
        
        # Load positive and negative tweets
        all_positive_tweets = twitter_samples.strings('positive_tweets.json')
        all_negative_tweets = twitter_samples.strings('negative_tweets.json')
        
        # Split data into train and test set (80/20 split)
        test_pos = all_positive_tweets[4000:]
        train_pos = all_positive_tweets[:4000]
        test_neg = all_negative_tweets[4000:]
        train_neg = all_negative_tweets[:4000]
        
        train_x = train_pos + train_neg 
        test_x = test_pos + test_neg
        
        # Create labels
        train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)), axis=0)
        test_y = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)), axis=0)
        
        print(f"Loaded Twitter samples: {len(train_x)} training, {len(test_x)} test samples")
        
        if save_to_file:
            self._save_twitter_data(train_x, test_x, train_y, test_y)
            
        return train_x, test_x, train_y, test_y
    
    def _save_twitter_data(self, train_x: List[str], test_x: List[str], 
                          train_y: np.ndarray, test_y: np.ndarray):
        """Save Twitter data to files"""
        # Save as JSON files
        twitter_data = {
            'train': {
                'texts': train_x,
                'labels': train_y.tolist()
            },
            'test': {
                'texts': test_x,
                'labels': test_y.tolist()
            }
        }
        
        with open(os.path.join(self.raw_data_path, 'twitter_samples.json'), 'w', encoding='utf-8') as f:
            json.dump(twitter_data, f, ensure_ascii=False, indent=2)
        
        # Save as pickle for easy loading
        with open(os.path.join(self.raw_data_path, 'twitter_samples.pkl'), 'wb') as f:
            pickle.dump((train_x, test_x, train_y, test_y), f)
            
        print(f"Twitter data saved to {self.raw_data_path}/")
    
    def load_huggingface_dataset(self, dataset_name: str = "stanfordnlp/sentiment140", 
                                save_to_file: bool = True) -> dict:
        """Load dataset from Hugging Face
        
        Args:
            dataset_name: Name of the Hugging Face dataset
            save_to_file: Whether to save the loaded data to files
            
        Returns:
            Dictionary containing the dataset
        """
        print(f"Loading {dataset_name} from Hugging Face...")
        
        try:
            dataset = load_dataset(dataset_name)
            print(f"Loaded {dataset_name} successfully")
            
            if save_to_file:
                self._save_huggingface_data(dataset, dataset_name)
                
            return dataset
            
        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
            return None
    
    def _save_huggingface_data(self, dataset, dataset_name: str):
        """Save Hugging Face dataset to files"""
        dataset_folder = os.path.join(self.raw_data_path, dataset_name.replace('/', '_'))
        os.makedirs(dataset_folder, exist_ok=True)
        
        # Save each split
        for split_name, split_data in dataset.items():
            # Convert to pandas DataFrame
            df = pd.DataFrame(split_data)
            
            # Save as CSV
            csv_path = os.path.join(dataset_folder, f'{split_name}.csv')
            df.to_csv(csv_path, index=False)
            
            # Save as JSON
            json_path = os.path.join(dataset_folder, f'{split_name}.json')
            df.to_json(json_path, orient='records', lines=True)
            
        print(f"Hugging Face data saved to {dataset_folder}/")
    
    def load_saved_twitter_data(self) -> Tuple[List[str], List[str], np.ndarray, np.ndarray]:
        """Load previously saved Twitter data
        
        Returns:
            Tuple of (train_x, test_x, train_y, test_y)
        """
        pkl_path = os.path.join(self.raw_data_path, 'twitter_samples.pkl')
        
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                return pickle.load(f)
        else:
            print("No saved Twitter data found. Loading fresh data...")
            return self.load_twitter_samples()
    
    def get_data_info(self) -> dict:
        """Get information about available datasets
        
        Returns:
            Dictionary with dataset information
        """
        info = {
            'raw_data_path': self.raw_data_path,
            'available_files': []
        }
        
        if os.path.exists(self.raw_data_path):
            for item in os.listdir(self.raw_data_path):
                item_path = os.path.join(self.raw_data_path, item)
                if os.path.isfile(item_path):
                    info['available_files'].append(item)
                elif os.path.isdir(item_path):
                    info['available_files'].append(f"{item}/ (directory)")
                    
        return info

# Convenience functions
def load_twitter_data(raw_data_path: str = "dataset/raw") -> Tuple[List[str], List[str], np.ndarray, np.ndarray]:
    """Convenience function to load Twitter data"""
    loader = DataLoader(raw_data_path)
    return loader.load_saved_twitter_data()

def load_sentiment140_data(raw_data_path: str = "dataset/raw") -> dict:
    """Convenience function to load Sentiment140 data"""
    loader = DataLoader(raw_data_path)
    return loader.load_huggingface_dataset("stanfordnlp/sentiment140")