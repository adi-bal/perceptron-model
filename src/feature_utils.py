"""Feature extraction utilities for text classification."""

from collections import defaultdict
from typing import Dict, List, Callable, Set


def create_bow_features(text: str) -> Dict[str, float]:
    """Create bag-of-words features from text.
    
    Args:
        text: Input text
        
    Returns:
        Dictionary mapping feature names to values
    """
    features = defaultdict(float)
    # Add bias term
    features["BIAS"] = 1.0
    
    # Add word features
    for word in text.lower().split():
        features[f"WORD_{word}"] = 1.0
        
    return dict(features)


def create_ngram_features(text: str, n: int = 2) -> Dict[str, float]:
    """Create n-gram features from text.
    
    Args:
        text: Input text
        n: Size of n-grams
        
    Returns:
        Dictionary mapping feature names to values
    """
    features = defaultdict(float)
    # Add bias term
    features["BIAS"] = 1.0
    
    # Add word features
    words = text.lower().split()
    
    # Add unigrams
    for word in words:
        features[f"WORD_{word}"] = 1.0
    
    # Add n-grams
    if n > 1:
        for i in range(len(words) - n + 1):
            ngram = "_".join(words[i:i+n])
            features[f"NGRAM_{ngram}"] = 1.0
            
    return dict(features)


def create_length_features(text: str) -> Dict[str, float]:
    """Create length-based features from text.
    
    Args:
        text: Input text
        
    Returns:
        Dictionary mapping feature names to values
    """
    features = defaultdict(float)
    
    # Add length features
    words = text.split()
    features["LENGTH_CHARS"] = len(text)
    features["LENGTH_WORDS"] = len(words)
    features["AVG_WORD_LENGTH"] = sum(len(w) for w in words) / max(1, len(words))
    
    return dict(features)


def combine_features(*feature_extractors: Callable[[str], Dict[str, float]]) -> Callable[[str], Dict[str, float]]:
    """Combine multiple feature extractors into one.
    
    Args:
        *feature_extractors: Functions that convert text to features
        
    Returns:
        Function that applies all extractors and combines the results
    """
    def combined_extractor(text: str) -> Dict[str, float]:
        combined = {}
        for extractor in feature_extractors:
            combined.update(extractor(text))
        return combined
    
    return combined_extractor


def featurize_texts(texts: List[str], featurizer: Callable[[str], Dict[str, float]]) -> List[Dict[str, float]]:
    """Convert raw texts to feature dictionaries.
    
    Args:
        texts: List of text strings
        featurizer: Function that converts text to features
        
    Returns:
        List of feature dictionaries
    """
    return [featurizer(text) for text in texts]