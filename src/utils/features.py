"""
Utility functions for sentiment analysis

This module contains helper functions for feature extraction, frequency building,
and other common operations used throughout the sentiment analysis pipeline.
"""

import numpy as np
import pickle
import json
import os
from typing import Dict, List, Tuple, Any, Union
from collections import defaultdict

def build_frequency_dict(tweets: List[str], labels: np.ndarray, 
                        process_tweet_func) -> Dict[Tuple[str, float], int]:
    """Build frequency dictionary from tweets and labels
    
    Args:
        tweets: List of tweet texts
        labels: Array of labels (0 or 1)
        process_tweet_func: Function to process tweets
        
    Returns:
        Dictionary mapping (word, label) pairs to frequencies
    """
    print("Building frequency dictionary...")
    
    labels_list = np.squeeze(labels).tolist()
    freqs = {}
    
    for label, tweet in zip(labels_list, tweets):
        processed_words = process_tweet_func(tweet)
        for word in processed_words:
            pair = (word, label)
            freqs[pair] = freqs.get(pair, 0) + 1
    
    print(f"Built frequency dictionary with {len(freqs)} word-label pairs")
    return freqs

def extract_features(tweet: str, freqs: Dict[Tuple[str, float], int], 
                    process_tweet_func) -> np.ndarray:
    """Extract features from a tweet
    
    Args:
        tweet: Tweet text
        freqs: Frequency dictionary
        process_tweet_func: Function to process tweets
        
    Returns:
        Feature vector [bias, pos_count, neg_count]
    """
    # Process tweet to get list of words
    word_list = process_tweet_func(tweet)
    
    # Initialize feature vector [bias, positive_count, negative_count]
    features = np.zeros((1, 3))
    
    # Set bias term to 1
    features[0, 0] = 1
    
    # Count positive and negative word frequencies
    for word in word_list:
        # Count positive occurrences
        if (word, 1) in freqs:
            features[0, 1] += freqs[(word, 1)]
        
        # Count negative occurrences  
        if (word, 0) in freqs:
            features[0, 2] += freqs[(word, 0)]
    
    return features

def build_feature_matrix(tweets: List[str], freqs: Dict[Tuple[str, float], int],
                        process_tweet_func) -> np.ndarray:
    """Build feature matrix for a list of tweets
    
    Args:
        tweets: List of tweet texts
        freqs: Frequency dictionary
        process_tweet_func: Function to process tweets
        
    Returns:
        Feature matrix (n_samples, 3)
    """
    print(f"Building feature matrix for {len(tweets)} tweets...")
    
    X = np.zeros((len(tweets), 3))
    
    for i, tweet in enumerate(tweets):
        X[i, :] = extract_features(tweet, freqs, process_tweet_func)
        
        # Progress indicator
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(tweets)} tweets")
    
    print(f"Feature matrix shape: {X.shape}")
    return X

def save_frequency_dict(freqs: Dict[Tuple[str, float], int], 
                       filepath: str) -> None:
    """Save frequency dictionary to file
    
    Args:
        freqs: Frequency dictionary
        filepath: Path to save the dictionary
    """
    # Convert tuple keys to strings for JSON serialization
    freqs_serializable = {f"{word}_{int(label)}": count 
                         for (word, label), count in freqs.items()}
    
    # Save as JSON
    json_path = filepath.replace('.pkl', '.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(freqs_serializable, f, ensure_ascii=False, indent=2)
    
    # Save as pickle (preserves original format)
    with open(filepath, 'wb') as f:
        pickle.dump(freqs, f)
    
    print(f"Frequency dictionary saved to {filepath} and {json_path}")

def load_frequency_dict(filepath: str) -> Dict[Tuple[str, float], int]:
    """Load frequency dictionary from file
    
    Args:
        filepath: Path to the saved dictionary
        
    Returns:
        Frequency dictionary
    """
    with open(filepath, 'rb') as f:
        freqs = pickle.load(f)
    
    print(f"Loaded frequency dictionary with {len(freqs)} entries")
    return freqs

def get_vocabulary_stats(freqs: Dict[Tuple[str, float], int]) -> Dict[str, Any]:
    """Get vocabulary statistics from frequency dictionary
    
    Args:
        freqs: Frequency dictionary
        
    Returns:
        Dictionary with vocabulary statistics
    """
    # Extract words and their frequencies
    words = set()
    pos_words = set()
    neg_words = set()
    pos_freq_total = 0
    neg_freq_total = 0
    
    for (word, label), count in freqs.items():
        words.add(word)
        if label == 1:
            pos_words.add(word)
            pos_freq_total += count
        else:
            neg_words.add(word)
            neg_freq_total += count
    
    stats = {
        'total_unique_words': len(words),
        'positive_words': len(pos_words),
        'negative_words': len(neg_words),
        'total_positive_frequency': pos_freq_total,
        'total_negative_frequency': neg_freq_total,
        'words_in_both_classes': len(pos_words.intersection(neg_words))
    }
    
    return stats

def get_top_words(freqs: Dict[Tuple[str, float], int], 
                 label: int = 1, top_n: int = 20) -> List[Tuple[str, int]]:
    """Get top words for a specific label
    
    Args:
        freqs: Frequency dictionary
        label: Label to get top words for (0 or 1)
        top_n: Number of top words to return
        
    Returns:
        List of (word, frequency) tuples sorted by frequency
    """
    # Filter words for the specified label
    label_words = {word: count for (word, lbl), count in freqs.items() if lbl == label}
    
    # Sort by frequency
    top_words = sorted(label_words.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    return top_words

def print_vocabulary_analysis(freqs: Dict[Tuple[str, float], int]) -> None:
    """Print detailed vocabulary analysis
    
    Args:
        freqs: Frequency dictionary
    """
    print("=" * 50)
    print("VOCABULARY ANALYSIS")
    print("=" * 50)
    
    # Get statistics
    stats = get_vocabulary_stats(freqs)
    
    print(f"Total unique words: {stats['total_unique_words']}")
    print(f"Words appearing in positive tweets: {stats['positive_words']}")
    print(f"Words appearing in negative tweets: {stats['negative_words']}")
    print(f"Words appearing in both classes: {stats['words_in_both_classes']}")
    print(f"Total positive word occurrences: {stats['total_positive_frequency']}")
    print(f"Total negative word occurrences: {stats['total_negative_frequency']}")
    print()
    
    # Top positive words
    print("Top 10 Positive Words:")
    top_pos = get_top_words(freqs, label=1, top_n=10)
    for i, (word, freq) in enumerate(top_pos, 1):
        print(f"{i:2d}. {word:15s} ({freq:4d} times)")
    print()
    
    # Top negative words
    print("Top 10 Negative Words:")
    top_neg = get_top_words(freqs, label=0, top_n=10)
    for i, (word, freq) in enumerate(top_neg, 1):
        print(f"{i:2d}. {word:15s} ({freq:4d} times)")

def create_train_test_split(X: np.ndarray, y: np.ndarray, 
                           test_size: float = 0.2, 
                           random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create train/test split from features and labels
    
    Args:
        X: Feature matrix
        y: Labels
        test_size: Fraction of data to use for testing
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    np.random.seed(random_seed)
    
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    
    # Create random indices
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    print(f"Created train/test split:")
    print(f"Training: {len(X_train)} samples")
    print(f"Testing: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test

def sigmoid(z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Sigmoid activation function
    
    Args:
        z: Input value(s)
        
    Returns:
        Sigmoid of z
    """
    # Clip z to prevent overflow
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def save_model_artifacts(model, freqs: Dict, processed_data: Dict, 
                        save_dir: str = "models") -> None:
    """Save all model artifacts for easy loading later
    
    Args:
        model: Trained model
        freqs: Frequency dictionary
        processed_data: Processed training/test data
        save_dir: Directory to save artifacts
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(save_dir, "sentiment_model.pkl")
    model.save_model(model_path)
    
    # Save frequency dictionary
    freqs_path = os.path.join(save_dir, "frequency_dict.pkl")
    save_frequency_dict(freqs, freqs_path)
    
    # Save processed data
    data_path = os.path.join(save_dir, "processed_data.pkl")
    with open(data_path, 'wb') as f:
        pickle.dump(processed_data, f)
    
    print(f"All model artifacts saved to {save_dir}/")

def load_model_artifacts(save_dir: str = "models") -> Tuple[Any, Dict, Dict]:
    """Load all model artifacts
    
    Args:
        save_dir: Directory containing saved artifacts
        
    Returns:
        Tuple of (model, freqs, processed_data)
    """
    from models.logistic_regression import LogisticRegressionModel
    
    # Load model
    model = LogisticRegressionModel()
    model_path = os.path.join(save_dir, "sentiment_model.pkl")
    model.load_model(model_path)
    
    # Load frequency dictionary
    freqs_path = os.path.join(save_dir, "frequency_dict.pkl")
    freqs = load_frequency_dict(freqs_path)
    
    # Load processed data
    data_path = os.path.join(save_dir, "processed_data.pkl")
    with open(data_path, 'rb') as f:
        processed_data = pickle.load(f)
    
    print(f"All model artifacts loaded from {save_dir}/")
    return model, freqs, processed_data