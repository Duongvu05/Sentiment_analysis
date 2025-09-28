"""
Logistic Regression model for sentiment analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Union, Tuple, List, Optional
from .base import SentimentModel

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

class LogisticRegressionModel(SentimentModel):
    """Logistic Regression model for sentiment analysis"""
    
    def __init__(self):
        super().__init__()
        self.weights = None
        self.loss_history = []
        
    def train(self, X: np.ndarray, y: np.ndarray, 
              learning_rate: float = 1e-9, 
              num_iterations: int = 10000,
              plot_loss: bool = True,
              verbose: bool = True) -> Dict[str, Any]:
        """Train logistic regression model using gradient descent
        
        Args:
            X: Feature matrix (m, n+1) where m is samples and n+1 includes bias
            y: Labels (m, 1)
            learning_rate: Learning rate for gradient descent
            num_iterations: Number of training iterations
            plot_loss: Whether to plot loss curve
            verbose: Whether to print training progress
            
        Returns:
            Dictionary with training results
        """
        # Initialize weights
        m, n = X.shape
        self.weights = np.zeros((n, 1))
        self.loss_history = []
        
        if verbose:
            print(f"Training Logistic Regression with {m} samples, {n} features")
            print(f"Learning rate: {learning_rate}, Iterations: {num_iterations}")
        
        # Gradient descent
        for i in range(num_iterations):
            # Forward pass
            z = np.dot(X, self.weights)
            predictions = sigmoid(z)
            
            # Compute loss (binary cross-entropy)
            # Add small epsilon to prevent log(0)
            epsilon = 1e-15
            predictions = np.clip(predictions, epsilon, 1 - epsilon)
            
            loss = -(np.dot(y.T, np.log(predictions)) + np.dot((1 - y).T, np.log(1 - predictions))) / m
            
            # Convert loss to scalar - handle different numpy array shapes
            if hasattr(loss, 'shape'):
                if loss.shape == (1, 1):
                    loss_scalar = float(loss[0, 0])
                elif loss.shape == ():
                    loss_scalar = float(loss)
                else:
                    loss_scalar = float(loss.flatten()[0])
            else:
                loss_scalar = float(loss)
                
            self.loss_history.append(loss_scalar)
            
            # Backward pass (compute gradients)
            dw = np.dot(X.T, (predictions - y)) / m
            
            # Update weights
            self.weights = self.weights - learning_rate * dw
            
            # Print progress
            if verbose and (i + 1) % 1000 == 0:
                print(f"Iteration {i + 1}/{num_iterations}, Loss: {loss_scalar:.8f}")
        
        final_loss = self.loss_history[-1]
        self.is_trained = True
        
        # Store model parameters
        self.model_params = {
            'weights': self.weights.copy(),
            'loss_history': self.loss_history.copy(),
            'learning_rate': learning_rate,
            'num_iterations': num_iterations,
            'final_loss': final_loss
        }
        
        if verbose:
            print(f"Training completed. Final loss: {final_loss:.8f}")
            print(f"Final weights: {[round(w, 8) for w in np.squeeze(self.weights)]}")
        
        # Plot loss curve
        if plot_loss and len(self.loss_history) > 1:
            self._plot_loss_curve()
        
        return {
            'final_loss': final_loss,
            'weights': self.weights.copy(),
            'loss_history': self.loss_history.copy(),
            'converged': True
        }
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities
        
        Args:
            X: Feature matrix or single feature vector
            
        Returns:
            Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Handle single sample
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        z = np.dot(X, self.weights)
        return sigmoid(z)
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Make binary predictions
        
        Args:
            X: Feature matrix or single feature vector
            threshold: Decision threshold
            
        Returns:
            Binary predictions (0 or 1)
        """
        probabilities = self.predict_proba(X)
        return (probabilities > threshold).astype(int)
    
    def predict_tweet(self, tweet_features: np.ndarray) -> float:
        """Predict sentiment for a single tweet
        
        Args:
            tweet_features: Feature vector for the tweet
            
        Returns:
            Prediction probability
        """
        return float(self.predict_proba(tweet_features.reshape(1, -1)))
    
    def _plot_loss_curve(self):
        """Plot the loss curve during training"""
        plt.figure(figsize=(10, 6))
        iterations = np.arange(1, len(self.loss_history) + 1)
        plt.plot(iterations, self.loss_history, color='blue', linewidth=2, label='Training Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss (Binary Cross-Entropy)')
        plt.title('Logistic Regression Training Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance (absolute weights)
        
        Returns:
            Feature importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        return np.abs(self.weights.flatten())
    
    def plot_decision_boundary(self, X: np.ndarray, y: np.ndarray, 
                              feature_names: Optional[List[str]] = None):
        """Plot 2D decision boundary (for 3D features including bias)
        
        Args:
            X: Feature matrix (should have 3 columns: bias, pos_freq, neg_freq)
            y: True labels
            feature_names: Names of features
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to plot decision boundary")
        
        if X.shape[1] != 3:
            print("Decision boundary plotting is only supported for 3D features (bias + 2 features)")
            return
        
        # Extract positive and negative frequency features (skip bias column)
        pos_freq = X[:, 1]
        neg_freq = X[:, 2]
        
        # Create mesh grid
        x1_min, x1_max = pos_freq.min() - 1000, pos_freq.max() + 1000
        x2_min, x2_max = neg_freq.min() - 1000, neg_freq.max() + 1000
        x1_mesh, x2_mesh = np.meshgrid(np.arange(x1_min, x1_max, 100),
                                       np.arange(x2_min, x2_max, 100))
        
        # Calculate decision boundary
        w = self.weights.flatten()
        z_mesh = w[0] + w[1] * x1_mesh + w[2] * x2_mesh
        
        # Plot
        fig = plt.figure(figsize=(12, 8))
        
        # 2D plot
        ax1 = fig.add_subplot(121)
        y_flat = y.flatten()
        
        # Plot data points
        pos_mask = y_flat == 1
        neg_mask = y_flat == 0
        
        ax1.scatter(pos_freq[pos_mask], neg_freq[pos_mask], 
                   c='green', marker='o', label='Positive', alpha=0.6)
        ax1.scatter(pos_freq[neg_mask], neg_freq[neg_mask], 
                   c='red', marker='x', label='Negative', alpha=0.6)
        
        # Plot decision boundary (z = 0)
        if w[2] != 0:  # Avoid division by zero
            x1_boundary = np.linspace(x1_min, x1_max, 100)
            x2_boundary = -(w[0] + w[1] * x1_boundary) / w[2]
            ax1.plot(x1_boundary, x2_boundary, 'k-', linewidth=2, 
                    label='Decision Boundary', alpha=0.8)
        
        ax1.set_xlabel('Positive Word Frequency')
        ax1.set_ylabel('Negative Word Frequency')
        ax1.set_title('2D Decision Boundary')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 3D surface plot
        try:
            from mpl_toolkits.mplot3d import Axes3D
            ax2 = fig.add_subplot(122, projection='3d')
            surface = ax2.plot_surface(x1_mesh, x2_mesh, z_mesh, 
                                      alpha=0.3, cmap='viridis')
            
            # Plot data points in 3D
            z_data = w[0] + w[1] * pos_freq + w[2] * neg_freq
            ax2.scatter(pos_freq[pos_mask], neg_freq[pos_mask], z_data[pos_mask], 
                       c='green', marker='o', s=50, label='Positive')
            ax2.scatter(pos_freq[neg_mask], neg_freq[neg_mask], z_data[neg_mask], 
                       c='red', marker='^', s=50, label='Negative')
            
            ax2.set_xlabel('Positive Word Frequency')
            ax2.set_ylabel('Negative Word Frequency')
            ax2.set_zlabel('Logit (z)')
            ax2.set_title('3D Decision Surface')
            ax2.legend()
        except ImportError:
            print("3D plotting not available, skipping 3D visualization")
        
        plt.tight_layout()
        plt.show()