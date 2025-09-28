"""
Models module

Contains model definitions for sentiment analysis:
- Logistic Regression model for binary sentiment classification
- Base classes for sentiment analysis models
- Model utilities and evaluation functions
"""

from .base import SentimentModel
from .logistic_regression import LogisticRegressionModel, sigmoid

__all__ = ['SentimentModel', 'LogisticRegressionModel', 'sigmoid']