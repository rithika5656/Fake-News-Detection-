"""
Fake News Detection - Prediction Script
Use trained model to classify news articles as fake or real
"""

import pickle
import re
import nltk
from nltk.corpus import stopwords
import os

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


def preprocess_text(text):
    """
    Preprocess text using the same method as training
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


def load_model_and_vectorizer():
    """
    Load the trained model and vectorizer from pickle files
    """
    if not os.path.exists('model.pkl'):
        print("Error: 'model.pkl' not found!")
        print("Please run 'python train_model.py' first to train the model.")
        return None, None
    
    if not os.path.exists('vectorizer.pkl'):
        print("Error: 'vectorizer.pkl' not found!")
        print("Please run 'python train_model.py' first to train the model.")
        return None, None
    
    print("Loading model...")
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    print("Loading vectorizer...")
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    return model, vectorizer


def predict_news(text, model, vectorizer):
    """
    Predict if a news article is fake or real
    """
    # Preprocess text
    processed_text = preprocess_text(text)
    
    # Vectorize
    text_vec = vectorizer.transform([processed_text])
    
    # Predict
    prediction = model.predict(text_vec)[0]
    confidence = model.predict_proba(text_vec)[0]
    
    return prediction, confidence


def main():
    """
    Main function for interactive prediction
    """
    print("="*60)
    print("FAKE NEWS DETECTION - PREDICTION")
    print("="*60)
    
    # Load model and vectorizer
    model, vectorizer = load_model_and_vectorizer()
    
    if model is None or vectorizer is None:
        return
    
    print("Model loaded successfully!\n")
    print("Instructions:")
    print("- Enter a news headline or article text to classify")
    print("- Type 'quit' or 'exit' to stop\n")
    
    # Interactive prediction loop
    while True:
        user_input = input("\nEnter news text (or 'quit' to exit): ").strip()
        
        if user_input.lower() in ['quit', 'exit']:
            print("Exiting... Thank you!")
            break
        
        if not user_input:
            print("Please enter some text!")
            continue
        
        # Make prediction
        try:
            prediction, confidence = predict_news(user_input, model, vectorizer)
            
            print("\n" + "-"*60)
            print("PREDICTION RESULT:")
            print("-"*60)
            
            # Map prediction (assuming 0=Real, 1=Fake)
            if prediction == 1:
                result = "FAKE NEWS"
                color_indicator = "⚠️"
            else:
                result = "REAL NEWS"
                color_indicator = "✓"
            
            print(f"{color_indicator} Classification: {result}")
            print(f"Confidence: {max(confidence)*100:.2f}%")
            print(f"  - Real: {confidence[0]*100:.2f}%")
            print(f"  - Fake: {confidence[1]*100:.2f}%")
            print("-"*60)
        
        except Exception as e:
            print(f"Error during prediction: {e}")
            print("Please try again with different text.")


if __name__ == "__main__":
    main()
