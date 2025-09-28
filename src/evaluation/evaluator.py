"""
Evaluation utilities for sentiment analysis models
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple, Optional, Union
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from models.logistic_regression import LogisticRegressionModel

class ModelEvaluator:
    """Class for evaluating sentiment analysis models"""
    
    def __init__(self):
        """Initialize evaluator"""
        pass
    
    def evaluate_model(self, model: LogisticRegressionModel,
                      X_test: np.ndarray, y_test: np.ndarray,
                      verbose: bool = True) -> Dict[str, Any]:
        """Comprehensive model evaluation
        
        Args:
            model: Trained model to evaluate
            X_test: Test features
            y_test: Test labels
            verbose: Whether to print results
            
        Returns:
            Dictionary with evaluation metrics
        """
        if verbose:
            print("=" * 50)
            print("MODEL EVALUATION")
            print("=" * 50)
        
        # Get predictions
        y_pred_proba = model.predict_proba(X_test).flatten()
        y_pred = model.predict(X_test).flatten()
        y_true = y_test.flatten()
        
        # Calculate metrics
        accuracy = self.calculate_accuracy(y_true, y_pred)
        precision, recall, f1 = self.calculate_precision_recall_f1(y_true, y_pred)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'true_labels': y_true
        }
        
        if verbose:
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            print()
        
        return results
    
    def calculate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate accuracy
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Accuracy score
        """
        return np.mean(y_true == y_pred)
    
    def calculate_precision_recall_f1(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
        """Calculate precision, recall, and F1-score
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Tuple of (precision, recall, f1_score)
        """
        # Calculate confusion matrix elements
        tp = np.sum((y_true == 1) & (y_pred == 1))  # True Positives
        fp = np.sum((y_true == 0) & (y_pred == 1))  # False Positives
        fn = np.sum((y_true == 1) & (y_pred == 0))  # False Negatives
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             labels: List[str] = ['Negative', 'Positive'],
                             title: str = 'Confusion Matrix') -> None:
        """Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Class labels for display
            title: Plot title
        """
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.show()
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=labels))
    
    def analyze_errors(self, model: LogisticRegressionModel,
                      test_texts: List[str], X_test: np.ndarray, y_test: np.ndarray,
                      process_tweet_func, max_errors: int = 10,
                      verbose: bool = True) -> Dict[str, Any]:
        """Analyze prediction errors
        
        Args:
            model: Trained model
            test_texts: Original test texts
            X_test: Test features
            y_test: Test labels
            process_tweet_func: Function to process tweets
            max_errors: Maximum number of errors to display
            verbose: Whether to print error analysis
            
        Returns:
            Dictionary with error analysis
        """
        if verbose:
            print("=" * 50)
            print("ERROR ANALYSIS")
            print("=" * 50)
        
        # Get predictions
        y_pred_proba = model.predict_proba(X_test).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        y_true = y_test.flatten()
        
        # Find errors
        error_indices = np.where(y_true != y_pred)[0]
        
        errors_info = {
            'total_errors': len(error_indices),
            'error_rate': len(error_indices) / len(y_true),
            'false_positives': [],
            'false_negatives': []
        }
        
        if verbose:
            print(f"Total errors: {len(error_indices)} out of {len(y_true)} samples")
            print(f"Error rate: {errors_info['error_rate']:.4f}")
            print()
        
        # Analyze errors
        errors_shown = 0
        for idx in error_indices[:max_errors]:
            original_tweet = test_texts[idx]
            processed_tweet = process_tweet_func(original_tweet)
            true_label = int(y_true[idx])
            pred_proba = y_pred_proba[idx]
            pred_label = int(y_pred[idx])
            
            error_info = {
                'original_tweet': original_tweet,
                'processed_tweet': processed_tweet,
                'true_label': true_label,
                'predicted_probability': pred_proba,
                'predicted_label': pred_label
            }
            
            # Categorize error
            if true_label == 0 and pred_label == 1:  # False Positive
                errors_info['false_positives'].append(error_info)
                error_type = "FALSE POSITIVE"
            else:  # False Negative
                errors_info['false_negatives'].append(error_info)
                error_type = "FALSE NEGATIVE"
            
            if verbose:
                print(f"Error #{errors_shown + 1} - {error_type}")
                print(f"Original: {original_tweet}")
                print(f"Processed: {' '.join(processed_tweet)}")
                print(f"True: {true_label}, Predicted: {pred_label} (prob: {pred_proba:.4f})")
                print("-" * 30)
            
            errors_shown += 1
            if errors_shown >= max_errors:
                break
        
        return errors_info
    
    def test_custom_tweets(self, model: LogisticRegressionModel,
                          custom_tweets: List[str],
                          extract_features_func,
                          freqs: Dict,
                          process_tweet_func,
                          verbose: bool = True) -> List[Dict[str, Any]]:
        """Test model on custom tweets
        
        Args:
            model: Trained model
            custom_tweets: List of custom tweets to test
            extract_features_func: Function to extract features
            freqs: Frequency dictionary
            process_tweet_func: Function to process tweets
            verbose: Whether to print results
            
        Returns:
            List of prediction results
        """
        if verbose:
            print("=" * 50)
            print("CUSTOM TWEET PREDICTIONS")
            print("=" * 50)
        
        results = []
        
        for tweet in custom_tweets:
            # Extract features
            features = extract_features_func(tweet, freqs, process_tweet_func)
            
            # Make prediction
            prob = float(model.predict_proba(features)[0])  # Convert to scalar
            prediction = 1 if prob > 0.5 else 0
            sentiment = "Positive" if prediction == 1 else "Negative"
            confidence = float(prob if prediction == 1 else (1 - prob))  # Convert to scalar
            
            result = {
                'tweet': tweet,
                'prediction': prediction,
                'probability': prob,
                'sentiment': sentiment,
                'confidence': confidence
            }
            results.append(result)
            
            if verbose:
                print(f"Tweet: {tweet}")
                print(f"Sentiment: {sentiment} (confidence: {confidence:.4f})")
                print("-" * 30)
        
        return results
    
    def plot_prediction_distribution(self, y_pred_proba: np.ndarray, y_true: np.ndarray,
                                   title: str = "Prediction Probability Distribution") -> None:
        """Plot distribution of prediction probabilities
        
        Args:
            y_pred_proba: Prediction probabilities
            y_true: True labels
            title: Plot title
        """
        plt.figure(figsize=(12, 4))
        
        # Separate probabilities by true class
        pos_probs = y_pred_proba[y_true == 1]
        neg_probs = y_pred_proba[y_true == 0]
        
        # Plot histograms
        plt.subplot(1, 2, 1)
        plt.hist(pos_probs, bins=30, alpha=0.7, color='green', label='Positive (True)')
        plt.hist(neg_probs, bins=30, alpha=0.7, color='red', label='Negative (True)')
        plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.8, label='Decision Threshold')
        plt.xlabel('Prediction Probability')
        plt.ylabel('Count')
        plt.title('Probability Distribution by True Class')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot box plots
        plt.subplot(1, 2, 2)
        plt.boxplot([neg_probs, pos_probs], labels=['Negative (True)', 'Positive (True)'])
        plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.8, label='Decision Threshold')
        plt.ylabel('Prediction Probability')
        plt.title('Probability Distribution Box Plot')
        plt.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

# Convenience functions
def evaluate_sentiment_model(model: LogisticRegressionModel,
                           X_test: np.ndarray, y_test: np.ndarray,
                           verbose: bool = True) -> Dict[str, Any]:
    """Convenience function to evaluate a sentiment model
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        verbose: Whether to print results
        
    Returns:
        Evaluation results
    """
    evaluator = ModelEvaluator()
    return evaluator.evaluate_model(model, X_test, y_test, verbose)

def predict_tweet_sentiment(model: LogisticRegressionModel,
                           tweet: str,
                           extract_features_func,
                           freqs: Dict,
                           process_tweet_func,
                           verbose: bool = True) -> Dict[str, Any]:
    """Predict sentiment for a single tweet
    
    Args:
        model: Trained model
        tweet: Tweet text
        extract_features_func: Function to extract features
        freqs: Frequency dictionary
        process_tweet_func: Function to process tweets
        verbose: Whether to print result
        
    Returns:
        Prediction result
    """
    # Extract features
    features = extract_features_func(tweet, freqs, process_tweet_func)
    
    # Make prediction
    prob = float(model.predict_proba(features)[0])
    prediction = 1 if prob > 0.5 else 0
    sentiment = "Positive" if prediction == 1 else "Negative"
    confidence = float(prob if prediction == 1 else (1 - prob))  # Convert to scalar
    
    result = {
        'tweet': tweet,
        'prediction': prediction,
        'probability': prob,
        'sentiment': sentiment,
        'confidence': confidence
    }
    
    if verbose:
        print(f"Tweet: {tweet}")
        print(f"Sentiment: {sentiment} (probability: {prob:.4f})")
    
    return result