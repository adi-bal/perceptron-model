# Simple Perceptron Text Classifier

An implementation of a simple perceptron model for text classification. 

## Installation

1. Clone this repository:
```bash
git clone 
cd perceptron-model
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Basic Usage

```python
from perceptron_model import PerceptronModel
from features import create_bow_features, featurize_texts

# Prepare your data
texts = [
    "I loved this movie",
    "This film was terrible",
    "Great acting but weak plot",
    "Highly recommend watching this"
]
labels = ["positive", "negative", "neutral", "positive"]

# Create features
X = featurize_texts(texts, create_bow_features)

# Train model
model = PerceptronModel()
model.fit(X, labels, num_epochs=5, lr=0.1)

# Make predictions
new_texts = ["This was amazing", "I hated it"]
new_X = featurize_texts(new_texts, create_bow_features)
predictions = [model.predict(features) for features in new_X]
print(predictions)  # ['positive', 'negative']

# Save trained model
model.save_weights("my_model.json")

# Load model
new_model = PerceptronModel()
new_model.load_weights("my_model.json")
```