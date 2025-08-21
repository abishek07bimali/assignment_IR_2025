import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
import pickle
import warnings
warnings.filterwarnings('ignore')

class DocumentClassifier:
    def __init__(self):
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
        
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = None
        self.label_encoder = LabelEncoder()
        self.models = {}
        
    def preprocess_text(self, text, use_stemming=False, use_lemmatization=True, remove_stopwords=True):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        tokens = word_tokenize(text)
        
        if remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        if use_lemmatization:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        elif use_stemming:
            tokens = [self.stemmer.stem(token) for token in tokens]
        
        return ' '.join(tokens)
    
    def load_data(self, filepath='data/documents.json'):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        texts = []
        labels = []
        
        for category, documents in data.items():
            for doc in documents:
                texts.append(doc['text'])
                labels.append(category)
        
        return texts, labels
    
    def prepare_features(self, texts, labels, vectorizer_type='tfidf', max_features=5000):
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        if vectorizer_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
        else:
            self.vectorizer = CountVectorizer(max_features=max_features, ngram_range=(1, 2))
        
        X = self.vectorizer.fit_transform(processed_texts)
        y = self.label_encoder.fit_transform(labels)
        
        return X, y
    
    def train_models(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print("\n" + "="*50)
        print("Training Logistic Regression Model")
        print("="*50)
        lr_model = LogisticRegression(max_iter=1000, random_state=42)
        lr_model.fit(X_train, y_train)
        lr_pred = lr_model.predict(X_test)
        
        print(f"Accuracy: {accuracy_score(y_test, lr_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, lr_pred, 
                                   target_names=self.label_encoder.classes_))
        
        cv_scores_lr = cross_val_score(lr_model, X, y, cv=5)
        print(f"Cross-validation scores: {cv_scores_lr}")
        print(f"Mean CV score: {cv_scores_lr.mean():.4f} (+/- {cv_scores_lr.std() * 2:.4f})")
        
        self.models['logistic_regression'] = lr_model
        
        print("\n" + "="*50)
        print("Training Naive Bayes Model")
        print("="*50)
        nb_model = MultinomialNB(alpha=0.1)
        nb_model.fit(X_train, y_train)
        nb_pred = nb_model.predict(X_test)
        
        print(f"Accuracy: {accuracy_score(y_test, nb_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, nb_pred, 
                                   target_names=self.label_encoder.classes_))
        
        cv_scores_nb = cross_val_score(nb_model, X, y, cv=5)
        print(f"Cross-validation scores: {cv_scores_nb}")
        print(f"Mean CV score: {cv_scores_nb.mean():.4f} (+/- {cv_scores_nb.std() * 2:.4f})")
        
        self.models['naive_bayes'] = nb_model
        
        return {
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test,
            'lr_pred': lr_pred, 'nb_pred': nb_pred
        }
    
    def predict(self, text, model_name='naive_bayes'):
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
        
        processed_text = self.preprocess_text(text)
        X = self.vectorizer.transform([processed_text])
        
        model = self.models[model_name]
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        
        label = self.label_encoder.inverse_transform([prediction])[0]
        
        prob_dict = {}
        for i, prob in enumerate(probabilities):
            category = self.label_encoder.inverse_transform([i])[0]
            prob_dict[category] = prob
        
        return {
            'prediction': label,
            'probabilities': prob_dict,
            'model_used': model_name
        }
    
    def save_model(self, filepath='models/'):
        os.makedirs(filepath, exist_ok=True)
        
        with open(os.path.join(filepath, 'vectorizer.pkl'), 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        with open(os.path.join(filepath, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        for name, model in self.models.items():
            with open(os.path.join(filepath, f'{name}_model.pkl'), 'wb') as f:
                pickle.dump(model, f)
        
        print(f"Models saved to {filepath}")
    
    def load_model(self, filepath='models/'):
        with open(os.path.join(filepath, 'vectorizer.pkl'), 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        with open(os.path.join(filepath, 'label_encoder.pkl'), 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        model_files = ['logistic_regression_model.pkl', 'naive_bayes_model.pkl']
        for model_file in model_files:
            model_path = os.path.join(filepath, model_file)
            if os.path.exists(model_path):
                model_name = model_file.replace('_model.pkl', '')
                with open(model_path, 'rb') as f:
                    self.models[model_name] = pickle.load(f)
        
        print(f"Models loaded from {filepath}")

def main():
    classifier = DocumentClassifier()
    
    print("Loading data...")
    texts, labels = classifier.load_data()
    print(f"Loaded {len(texts)} documents")
    
    print("\nPreparing features...")
    X, y = classifier.prepare_features(texts, labels)
    print(f"Feature matrix shape: {X.shape}")
    
    print("\nTraining models...")
    results = classifier.train_models(X, y)
    
    print("\nSaving models...")
    classifier.save_model()
    
    print("\n" + "="*50)
    print("Testing with sample inputs")
    print("="*50)
    
    test_samples = [
        "The president announced new policies regarding healthcare reform.",
        "Stock market reached all-time highs following tech earnings.",
        "New study shows benefits of exercise for mental health.",
        "Election results spark protests in the capital city.",
        "Company merger creates largest retail chain in the country.",
        "Vaccine development shows promising results in trials."
    ]
    
    for sample in test_samples:
        print(f"\nInput: {sample[:80]}...")
        for model_name in ['logistic_regression', 'naive_bayes']:
            result = classifier.predict(sample, model_name)
            print(f"{model_name}: {result['prediction']} (confidence: {result['probabilities'][result['prediction']]:.2%})")

if __name__ == "__main__":
    main()