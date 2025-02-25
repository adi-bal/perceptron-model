"""Perceptron model for text classification with simplified API."""

import json
import os
from collections import defaultdict
from typing import Dict, List, Set, Optional
import math


class PerceptronModel:
    """Perceptron model for classification with simplified API."""

    def __init__(self):
        self.weights: Dict[str, float] = defaultdict(float)
        self.labels: Set[str] = set()
        
    def _get_weight_key(self, feature: str, label: str) -> str:
        """An internal hash function to build keys of self.weights."""
        return feature + "#" + str(label)

    def score(self, features: Dict[str, float], label: str) -> float:
        """Compute the score of a class given input features.

        Args:
            features: Dictionary of feature names to values
            label: Class label

        Returns:
            The output score
        """
        return sum(self.weights.get(self._get_weight_key(feature, label), 0.0) * value 
                  for feature, value in features.items())

    def predict(self, features: Dict[str, float]) -> str:
        """Predicts a label for input features.

        Args:
            features: Dictionary of feature names to values

        Returns:
            The predicted class
        """
        if not self.labels:
            return "unknown"  # Return default if no labels seen yet
        return max(self.labels, key=lambda label: self.score(features, label))

    def update_parameters(
        self, features: Dict[str, float], true_label: str, predicted_label: str, lr: float
    ) -> None:
        """Update the model weights using the perceptron update rule.

        Args:
            features: Dictionary of feature names to values
            true_label: The correct label
            predicted_label: The predicted label
            lr: Learning rate
        """
        if predicted_label == true_label:
            return  # No update needed if prediction is correct
        
        # Update weights for the correct label (add)
        for feature, value in features.items():
            weight_key = self._get_weight_key(feature, true_label)
            self.weights[weight_key] += lr * value
            
        # Update weights for the predicted label (subtract)
        for feature, value in features.items():
            weight_key = self._get_weight_key(feature, predicted_label)
            self.weights[weight_key] -= lr * value

    def fit(
        self,
        X_train: List[Dict[str, float]],
        y_train: List[str],
        X_val: Optional[List[Dict[str, float]]] = None,
        y_val: Optional[List[str]] = None,
        num_epochs: int = 3,
        lr: float = 0.1,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """Train the perceptron model on feature dictionaries and labels.

        Args:
            X_train: List of feature dictionaries for training
            y_train: List of true labels for training
            X_val: Optional list of feature dictionaries for validation
            y_val: Optional list of true labels for validation
            num_epochs: Number of training epochs
            lr: Learning rate
            verbose: Whether to print progress

        Returns:
            Dictionary with training and validation metrics
        """
        if len(X_train) != len(y_train):
            raise ValueError("X_train and y_train must have the same length")
            
        if X_val is not None and y_val is not None and len(X_val) != len(y_val):
            raise ValueError("X_val and y_val must have the same length")
            
        # Collect all possible labels
        self.labels.update(y_train)
            
        # Track metrics
        metrics = {
            "train_accuracy": [],
            "val_accuracy": []
        }
        
        # Train for num_epochs
        for epoch in range(num_epochs):
            # Training loop
            for features, true_label in zip(X_train, y_train):
                predicted_label = self.predict(features)
                self.update_parameters(features, true_label, predicted_label, lr)
            
            # Calculate training accuracy
            train_predictions = [self.predict(features) for features in X_train]
            train_acc = sum(p == t for p, t in zip(train_predictions, y_train)) / len(y_train)
            metrics["train_accuracy"].append(train_acc)
            
            # Calculate validation accuracy if provided
            if X_val is not None and y_val is not None:
                val_predictions = [self.predict(features) for features in X_val]
                val_acc = sum(p == t for p, t in zip(val_predictions, y_val)) / len(y_val)
                metrics["val_accuracy"].append(val_acc)
                
            # Print progress
            if verbose:
                epoch_str = f"Epoch {epoch + 1}/{num_epochs}, Train accuracy: {train_acc:.3f}"
                if X_val is not None and y_val is not None:
                    epoch_str += f", Validation accuracy: {metrics['val_accuracy'][-1]:.3f}"
                print(epoch_str)
                
        return metrics

    def evaluate(self, X: List[Dict[str, float]], y: List[str]) -> float:
        """Evaluate the model on feature dictionaries and labels.

        Args:
            X: List of feature dictionaries
            y: List of true labels

        Returns:
            Accuracy as a float between 0 and 1
        """
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
            
        if not X:
            return 0.0
            
        predictions = [self.predict(features) for features in X]
        return sum(p == t for p, t in zip(predictions, y)) / len(y)
    
    def predict_proba(self, features: Dict[str, float]) -> Dict[str, float]:
        """Get probability-like scores for all labels.

        This isn't a true probability, but rather normalized scores.

        Args:
            features: Dictionary of feature names to values

        Returns:
            Dictionary mapping labels to scores
        """
        if not self.labels:
            return {}
            
        # Get raw scores
        scores = {label: self.score(features, label) for label in self.labels}
        
        # Convert to "probability-like" values with softmax
        max_score = max(scores.values())
        exp_scores = {label: math.exp(score - max_score) for label, score in scores.items()}
        total = sum(exp_scores.values())
        
        return {label: score/total for label, score in exp_scores.items()} if total > 0 else scores
                
    def save_weights(self, path: str) -> None:
        """Save model weights to a JSON file.
        
        Args:
            path: Path to save the weights
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(json.dumps(self.weights, indent=2, sort_keys=True))
            
    def load_weights(self, path: str) -> None:
        """Load model weights from a JSON file.
        
        Args:
            path: Path to the weights file
        """
        with open(path, "r") as f:
            self.weights = json.load(f)

        self.labels = set()
        for key in self.weights:
            if "#" in key:
                label = key.split("#")[1]
                self.labels.add(label)