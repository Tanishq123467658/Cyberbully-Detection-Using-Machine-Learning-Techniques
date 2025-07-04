import os
import argparse
import nltk
import pickle

# Download required NLTK resources
print("Downloading required NLTK resources...")
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from preprocess import load_and_preprocess_data, TextPreprocessor
from feature_extraction import prepare_data
from model import train_and_evaluate_model, CyberbullyingClassifier
from visualization import visualize_data

def test_model_with_text(text, model_path='models/logistic_model.pkl', 
                        feature_extractor_path='models/feature_extractor_tfidf.pkl'):
    """
    Test the trained model with a new text input
    
    Parameters:
    -----------
    text : str
        Text to classify
    model_path : str
        Path to the trained model
    feature_extractor_path : str
        Path to the feature extractor
    """
    # Load the model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load the feature extractor
    with open(feature_extractor_path, 'rb') as f:
        vectorizer = pickle.load(f)
    
    # Preprocess the text
    preprocessor = TextPreprocessor()
    processed_text = preprocessor.preprocess(text)
    
    # Transform the text to features
    features = vectorizer.transform([processed_text])
    
    # Make prediction
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    
    # Print result
    result = "Cyberbullying" if prediction == 1 else "Not Cyberbullying"
    print(f"\nInput text: {text}")
    print(f"Prediction: {result}")
    print(f"Confidence: {probability:.2f}")
    
    return result, probability

def main():
    parser = argparse.ArgumentParser(description='Cyberbullying Detection')
    parser.add_argument('--data', type=str, default='cyberbullying_tweets.csv', help='Path to dataset')
    parser.add_argument('--feature_method', type=str, default='tfidf', choices=['tfidf', 'count'], help='Feature extraction method')
    parser.add_argument('--model_type', type=str, default='logistic', choices=['logistic', 'rf', 'svm', 'gb'], help='Model type')
    parser.add_argument('--use_smote', action='store_true', help='Use SMOTE for oversampling')
    parser.add_argument('--tune_params', action='store_true', help='Tune hyperparameters')
    parser.add_argument('--visualize_only', action='store_true', help='Only generate visualizations')
    parser.add_argument('--test', action='store_true', help='Test the model with custom input')
    parser.add_argument('--text', type=str, help='Text to classify (used with --test)')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Test mode
    if args.test:
        if args.text:
            test_model_with_text(args.text, 
                               f'models/{args.model_type}_model.pkl',
                               f'models/feature_extractor_{args.feature_method}.pkl')
        else:
            # Interactive mode
            print("\nEnter text to classify (type 'exit' to quit):")
            while True:
                text = input("\nText: ")
                if text.lower() == 'exit':
                    break
                test_model_with_text(text, 
                                   f'models/{args.model_type}_model.pkl',
                                   f'models/feature_extractor_{args.feature_method}.pkl')
        return
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data(args.data)
    
    # Generate visualizations
    print("\nGenerating data visualizations...")
    visualize_data(df)
    
    if not args.visualize_only:
        # Prepare data
        print("\nPreparing data for model training...")
        data = prepare_data(df, feature_method=args.feature_method)
        
        # Train and evaluate model
        print(f"\nTraining and evaluating {args.model_type} model...")
        classifier, results = train_and_evaluate_model(
            data, 
            model_type=args.model_type,
            use_smote=args.use_smote,
            tune_params=args.tune_params
        )
        
        print("\nDone! Results and models saved to 'results' and 'models' directories.")
        
        # Test the model with a sample text
        print("\nTesting the model with a sample text:")
        test_model_with_text("You are so stupid and ugly, nobody likes you", 
                           f'models/{args.model_type}_model.pkl',
                           f'models/feature_extractor_{args.feature_method}.pkl')
        
        # Ask if user wants to test with custom input
        print("\nDo you want to test the model with custom input? (y/n)")
        if input().lower() == 'y':
            print("\nEnter text to classify (type 'exit' to quit):")
            while True:
                text = input("\nText: ")
                if text.lower() == 'exit':
                    break
                test_model_with_text(text, 
                                   f'models/{args.model_type}_model.pkl',
                                   f'models/feature_extractor_{args.feature_method}.pkl')

if __name__ == "__main__":
    main()