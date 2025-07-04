import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text):
        """Clean and preprocess text data"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags symbol (keep the text)
        text = re.sub(r'#', '', text)
        
        # Remove punctuation and special characters
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_lemmatize(self, text):
        """Tokenize and lemmatize text"""
        # Use simple split instead of word_tokenize to avoid punkt_tab dependency
        tokens = text.split()
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        return ' '.join(tokens)
    
    def preprocess(self, text):
        """Apply full preprocessing pipeline"""
        text = self.clean_text(text)
        text = self.tokenize_and_lemmatize(text)
        return text

def load_and_preprocess_data(file_path):
    """Load and preprocess the dataset"""
    # Load data
    df = pd.read_csv(file_path)
    
    # Check for missing values
    print(f"Missing values before cleaning: {df.isnull().sum()}")
    
    # Drop rows with missing values
    df = df.dropna()
    
    # Create binary labels (1 for cyberbullying, 0 for not_cyberbullying)
    df['label'] = df['cyberbullying_type'].apply(lambda x: 0 if x == 'not_cyberbullying' else 1)
    
    # Preprocess text
    preprocessor = TextPreprocessor()
    df['processed_text'] = df['tweet_text'].apply(preprocessor.preprocess)
    
    # Remove empty texts after preprocessing
    df = df[df['processed_text'] != '']
    
    print(f"Dataset shape after preprocessing: {df.shape}")
    print(f"Class distribution:\n{df['label'].value_counts()}")
    
    return df

if __name__ == "__main__":
    # Test the preprocessing
    df = load_and_preprocess_data("cyberbullying_tweets.csv")
    print("\nSample processed texts:")
    for i in range(min(5, len(df))):
        print(f"Original: {df['tweet_text'].iloc[i]}")
        print(f"Processed: {df['processed_text'].iloc[i]}")
        print(f"Label: {'Cyberbullying' if df['label'].iloc[i] == 1 else 'Not Cyberbullying'}")
        print("-" * 50)