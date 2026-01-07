# Fake News Detection (Basic ML)

A machine learning project to classify news articles as fake or real using NLP and Scikit-learn.

## Features

- **NLP-based Classification**: Uses TF-IDF vectorization for text processing
- **Machine Learning**: Implements Logistic Regression classifier
- **Dataset**: Uses pre-trained fake news dataset
- **Model Evaluation**: Includes accuracy, precision, recall, and F1-score metrics

## Tech Stack

- Python 3.x
- Scikit-learn - Machine Learning
- NLTK - Natural Language Processing
- Pandas - Data manipulation
- NumPy - Numerical computing

## Installation

1. Clone the repository:
```bash
git clone https://github.com/rithika5656/Fake-News-Detection-.git
cd Fake-News-Detection-
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK data (required for stopwords):
```python
python -c "import nltk; nltk.download('stopwords')"
```

## Project Structure

```
├── data/
│   └── news.csv              # Dataset file (download required)
├── train_model.py            # Model training script
├── predict.py                # Prediction script
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## Dataset

Download the fake news dataset from Kaggle:
- [Fake News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

Place the dataset in the `data/` folder as `news.csv`

## Usage

### 1. Train the Model
```bash
python train_model.py
```

This will:
- Load and preprocess the dataset
- Split into training and testing sets (80-20)
- Train the Logistic Regression model
- Save the trained model and vectorizer
- Display evaluation metrics

### 2. Make Predictions
```bash
python predict.py
```

Input a news headline/text to classify it as Fake or Real.

## Model Performance

The model typically achieves:
- **Accuracy**: ~95%
- **Precision**: ~94%
- **Recall**: ~96%
- **F1-Score**: ~95%

(Exact metrics depend on the dataset split)

## How It Works

1. **Data Preprocessing**: 
   - Converts text to lowercase
   - Removes punctuation and special characters
   - Removes stopwords using NLTK

2. **Feature Extraction**:
   - Uses TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer
   - Converts text into numerical features

3. **Classification**:
   - Trains Logistic Regression classifier
   - Evaluates using standard ML metrics

4. **Prediction**:
   - Takes new text input
   - Preprocesses and vectorizes it
   - Predicts fake or real classification

## Key Files

- `train_model.py` - Main training script that creates and saves the model
- `predict.py` - Script for making predictions on new articles
- `model.pkl` - Trained Logistic Regression model (auto-generated)
- `vectorizer.pkl` - Fitted TF-IDF vectorizer (auto-generated)

## Future Improvements

- Add more advanced NLP preprocessing (lemmatization, POS tagging)
- Experiment with ensemble methods (Random Forest, Gradient Boosting)
- Implement deep learning models (LSTM, BERT)
- Add web interface using Flask/Django
- Deploy as API service

## License

This project is open source and available under the MIT License.

## Author

Created as part of ML project portfolio.
