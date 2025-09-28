"""
Training utilities for sentiment analysis models
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, Optional, List
import os
from models.logistic_regression import LogisticRegressionModel

class ModelTrainer:
    """Class for training sentiment analysis models"""
    
    def __init__(self, model_save_path: str = "models"):
        """Initialize trainer
        
        Args:
            model_save_path: Path to save trained models
        """
        self.model_save_path = model_save_path
        os.makedirs(model_save_path, exist_ok=True)
        
    def train_logistic_regression(self, X: np.ndarray, y: np.ndarray,
                                 learning_rate: float = 1e-9,
                                 num_iterations: int = 10000,
                                 plot_loss: bool = True,
                                 verbose: bool = True,
                                 save_model: bool = True,
                                 model_name: str = "logistic_regression") -> LogisticRegressionModel:
        """Train a logistic regression model
        
        Args:
            X: Feature matrix (m, n+1) 
            y: Labels (m, 1)
            learning_rate: Learning rate for gradient descent
            num_iterations: Number of training iterations
            plot_loss: Whether to plot training loss
            verbose: Whether to print training progress
            save_model: Whether to save the trained model
            model_name: Name for saving the model
            
        Returns:
            Trained LogisticRegressionModel
        """
        if verbose:
            print("=" * 50)
            print("TRAINING LOGISTIC REGRESSION MODEL")
            print("=" * 50)
        
        # Initialize model
        model = LogisticRegressionModel()
        
        # Train the model
        training_results = model.train(
            X=X, 
            y=y,
            learning_rate=learning_rate,
            num_iterations=num_iterations,
            plot_loss=plot_loss,
            verbose=verbose
        )
        
        # Save model if requested
        if save_model:
            model_path = os.path.join(self.model_save_path, f"{model_name}.pkl")
            model.save_model(model_path)
            
            # Also save weights in numpy format for compatibility
            weights_path = os.path.join(self.model_save_path, f"{model_name}_weights.npy")
            np.save(weights_path, model.weights)
            
            if verbose:
                print(f"Model saved to: {model_path}")
                print(f"Weights saved to: {weights_path}")
        
        return model
    
    def load_model(self, model_path: str) -> LogisticRegressionModel:
        """Load a trained model
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Loaded LogisticRegressionModel
        """
        model = LogisticRegressionModel()
        model.load_model(model_path)
        return model
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray,
                      k_folds: int = 5,
                      learning_rate: float = 1e-9,
                      num_iterations: int = 10000,
                      verbose: bool = True) -> Dict[str, Any]:
        """Perform k-fold cross validation
        
        Args:
            X: Feature matrix
            y: Labels
            k_folds: Number of folds
            learning_rate: Learning rate for training
            num_iterations: Number of training iterations
            verbose: Whether to print progress
            
        Returns:
            Cross validation results
        """
        if verbose:
            print(f"Performing {k_folds}-fold cross validation...")
        
        n_samples = X.shape[0]
        fold_size = n_samples // k_folds
        
        fold_losses = []
        fold_accuracies = []
        
        for fold in range(k_folds):
            if verbose:
                print(f"\nFold {fold + 1}/{k_folds}")
            
            # Create train/validation split
            start_idx = fold * fold_size
            end_idx = (fold + 1) * fold_size if fold < k_folds - 1 else n_samples
            
            # Validation set
            X_val = X[start_idx:end_idx]
            y_val = y[start_idx:end_idx]
            
            # Training set (everything except validation)
            X_train = np.vstack([X[:start_idx], X[end_idx:]])
            y_train = np.vstack([y[:start_idx], y[end_idx:]])
            
            # Train model
            model = LogisticRegressionModel()
            training_results = model.train(
                X=X_train,
                y=y_train,
                learning_rate=learning_rate,
                num_iterations=num_iterations,
                plot_loss=False,
                verbose=False
            )
            
            # Evaluate on validation set
            val_accuracy = self._calculate_accuracy(model, X_val, y_val)
            
            fold_losses.append(training_results['final_loss'])
            fold_accuracies.append(val_accuracy)
            
            if verbose:
                print(f"Validation accuracy: {val_accuracy:.4f}")
        
        results = {
            'mean_loss': np.mean(fold_losses),
            'std_loss': np.std(fold_losses),
            'mean_accuracy': np.mean(fold_accuracies),
            'std_accuracy': np.std(fold_accuracies),
            'fold_losses': fold_losses,
            'fold_accuracies': fold_accuracies
        }
        
        if verbose:
            print(f"\nCross Validation Results:")
            print(f"Mean Accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
            print(f"Mean Loss: {results['mean_loss']:.6f} ± {results['std_loss']:.6f}")
        
        return results
    
    def _calculate_accuracy(self, model: LogisticRegressionModel, 
                           X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy for a model
        
        Args:
            model: Trained model
            X: Feature matrix
            y: True labels
            
        Returns:
            Accuracy score
        """
        predictions = model.predict(X)
        return np.mean(predictions.flatten() == y.flatten())

def train_sentiment_model(X_train: np.ndarray, y_train: np.ndarray,
                         learning_rate: float = 1e-9,
                         num_iterations: int = 10000,
                         verbose: bool = True) -> LogisticRegressionModel:
    """Convenience function to train a sentiment analysis model
    
    Args:
        X_train: Training features
        y_train: Training labels
        learning_rate: Learning rate
        num_iterations: Number of iterations
        verbose: Whether to print progress
        
    Returns:
        Trained model
    """
    trainer = ModelTrainer()
    return trainer.train_logistic_regression(
        X=X_train,
        y=y_train,
        learning_rate=learning_rate,
        num_iterations=num_iterations,
        verbose=verbose
    )