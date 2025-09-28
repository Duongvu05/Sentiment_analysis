"""
Base model class for sentiment analysis
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union, Tuple

class SentimentModel(ABC):
    """Abstract base class for sentiment analysis models"""
    
    def __init__(self):
        self.is_trained = False
        self.model_params = {}
        
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Train the model
        
        Args:
            X: Feature matrix
            y: Labels
            **kwargs: Additional training parameters
            
        Returns:
            Training results dictionary
        """
        pass
    
    @abstractmethod
    def predict(self, X: Union[np.ndarray, str]) -> Union[float, np.ndarray]:
        """Make predictions
        
        Args:
            X: Input data (features or raw text)
            
        Returns:
            Prediction(s)
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: Union[np.ndarray, str]) -> Union[float, np.ndarray]:
        """Get prediction probabilities
        
        Args:
            X: Input data (features or raw text)
            
        Returns:
            Prediction probability(ies)
        """
        pass
    
    def save_model(self, filepath: str):
        """Save model to file
        
        Args:
            filepath: Path to save the model
        """
        import pickle
        
        model_data = {
            'model_params': self.model_params,
            'is_trained': self.is_trained,
            'model_type': self.__class__.__name__
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load model from file
        
        Args:
            filepath: Path to the saved model
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model_params = model_data['model_params']
        self.is_trained = model_data['is_trained']