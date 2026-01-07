"""
Fake News Detection - Model Training Script
Trains a Logistic Regression model to classify news as fake or real
"""

import pandas as pd
import numpy as np
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def preprocess_text(text):
    """
    Preprocess text by removing special characters, converting to lowercase,
    and removing stopwords
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    
    return ' '.join(tokens)


def load_and_prepare_data(filepath):
    """
    Load dataset and prepare for training
    """
    print("Loading dataset...")
    
    try:
        df = pd.read_csv(filepath)
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Ensure we have required columns
        if 'text' not in df.columns and 'title' in df.columns:
            df['text'] = df['title'].fillna('') + ' ' + df.get('text', '').fillna('')
        
        if 'label' not in df.columns:
            if 'class' in df.columns:
                df['label'] = df['class']
            else:
                print("Error: No label column found")
                return None, None
        
        # Remove duplicates and missing values
        df = df.dropna(subset=['text', 'label'])
        df = df.drop_duplicates(subset=['text'])
        
        print(f"Dataset after cleaning: {df.shape}")
        print(f"\nLabel distribution:\n{df['label'].value_counts()}")
        
        return df['text'], df['label']
    
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found")
        print("Please download the dataset from Kaggle:")
        print("https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset")
        print("\nPlace the CSV file in the data/ folder as 'news.csv'")
        return None, None


def train_model():
    """
    Main function to train the fake news detection model
    """
    # Load data
    X, y = load_and_prepare_data('data/news.csv')
    
    if X is None or y is None:
        print("Failed to load data. Exiting...")
        return
    
    # Preprocess text
    print("\nPreprocessing text...")
    X_processed = X.apply(preprocess_text)
    
    # Split data
    print("Splitting data into training and testing sets (80-20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
    
    # Vectorize text using TF-IDF
    print("\nVectorizing text using TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=5,
        max_df=0.8,
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print(f"Feature matrix shape: {X_train_vec.shape}")
    
    # Train Logistic Regression model
    print("\nTraining Logistic Regression model...")
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train_vec, y_train)
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred = model.predict(X_test_vec)
    
    # Evaluate model
    print("\n" + "="*50)
    print("MODEL PERFORMANCE")
    print("="*50)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:\n{cm}")
    
    # Save model and vectorizer
    print("\nSaving model and vectorizer...")
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print("Model saved as 'model.pkl'")
    print("Vectorizer saved as 'vectorizer.pkl'")
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    train_model()
