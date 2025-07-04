import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
import pickle
import os

class FeatureExtractor:
    def __init__(self, method='tfidf', max_features=5000):
        """
        Initialize feature extractor
        
        Parameters:
        -----------
        method : str
            Feature extraction method ('tfidf', 'count')
        max_features : int
            Maximum number of features for TF-IDF or Count vectorizer
        """
        self.method = method
        self.max_features = max_features
        
        if method == 'tfidf':
            self.vectorizer = TfidfVectorizer(max_features=max_features, 
                                             min_df=5, 
                                             max_df=0.8,
                                             ngram_range=(1, 2))
        elif method == 'count':
            self.vectorizer = CountVectorizer(max_features=max_features,
                                             min_df=5,
                                             max_df=0.8,
                                             ngram_range=(1, 2))
        else:
            raise ValueError(f"Unsupported method: {method}. Choose from 'tfidf' or 'count'")
    
    def fit_transform(self, texts):
        """Fit and transform texts to features"""
        return self.vectorizer.fit_transform(texts)
    
    def transform(self, texts):
        """Transform texts to features"""
        return self.vectorizer.transform(texts)
    
    def save(self, file_path):
        """Save the feature extractor"""
        with open(file_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
    
    def load(self, file_path):
        """Load the feature extractor"""
        with open(file_path, 'rb') as f:
            self.vectorizer = pickle.load(f)

def prepare_data(df, feature_method='tfidf', test_size=0.2, val_size=0.15, random_state=42):
    """
    Prepare data for model training
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Preprocessed dataframe
    feature_method : str
        Feature extraction method
    test_size : float
        Proportion of data to use for testing
    val_size : float
        Proportion of training data to use for validation
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    dict
        Dictionary containing train, validation, and test data
    """
    # Split data into train and test
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['label'])
    
    # Split train into train and validation
    train_df, val_df = train_test_split(train_df, test_size=val_size, random_state=random_state, stratify=train_df['label'])
    
    print(f"Train set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"Test set size: {len(test_df)}")
    
    # Extract features
    feature_extractor = FeatureExtractor(method=feature_method)
    
    X_train = feature_extractor.fit_transform(train_df['processed_text'])
    X_val = feature_extractor.transform(val_df['processed_text'])
    X_test = feature_extractor.transform(test_df['processed_text'])
    
    y_train = train_df['label'].values
    y_val = val_df['label'].values
    y_test = test_df['label'].values
    
    # Save feature extractor
    os.makedirs('models', exist_ok=True)
    feature_extractor.save(f'models/feature_extractor_{feature_method}.pkl')
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'feature_extractor': feature_extractor,
        'train_df': train_df,
        'val_df': val_df,
        'test_df': test_df
    }

if __name__ == "__main__":
    from preprocess import load_and_preprocess_data
    
    # Load and preprocess data
    df = load_and_preprocess_data("cyberbullying_tweets.csv")
    
    # Prepare data
    data = prepare_data(df, feature_method='tfidf')
    
    # Print feature information
    if hasattr(data['feature_extractor'], 'vectorizer'):
        print(f"Number of features: {len(data['feature_extractor'].vectorizer.get_feature_names_out())}")
        print("Top 20 features:")
        feature_names = data['feature_extractor'].vectorizer.get_feature_names_out()
        print(feature_names[:20])