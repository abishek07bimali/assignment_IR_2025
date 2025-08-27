#!/usr/bin/env python3
"""
Text Classification Engine
Author: Abishek Bimali
Date: 2025
"""

import json
import pickle
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import nltk

warnings.filterwarnings('ignore')


class TextPreprocessor:
    """Handles text cleaning and preprocessing"""
    
    def __init__(self):
        self._setup_nltk()
        self.word_tokenizer = nltk.word_tokenize
        self.sent_tokenizer = nltk.sent_tokenize
        self.lemmatizer = nltk.WordNetLemmatizer()
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
    
    def _setup_nltk(self):
        """Download required NLTK resources"""
        resources = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
        for resource in resources:
            try:
                nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' 
                             else f'corpora/{resource}')
            except LookupError:
                nltk.download(resource, quiet=True)
    
    def clean_text(self, text: str) -> str:
        """Basic text cleaning"""
        # Convert to lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        # Remove extra spaces
        text = ' '.join(text.split())
        return text
    
    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        """Tokenize and lemmatize text"""
        tokens = self.word_tokenizer(text)
        # Keep only alphabetic tokens
        tokens = [token for token in tokens if token.isalpha()]
        # Remove stopwords
        tokens = [token for token in tokens if token not in self.stop_words]
        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return tokens
    
    def process(self, text: str) -> str:
        """Full preprocessing pipeline"""
        text = self.clean_text(text)
        tokens = self.tokenize_and_lemmatize(text)
        return ' '.join(tokens)


class DocumentAnalyzer:
    """Analyzes document characteristics"""
    
    @staticmethod
    def extract_statistics(text: str) -> Dict:
        """Extract statistical features from text"""
        words = text.split()
        sentences = nltk.sent_tokenize(text)
        
        stats = {
            'num_words': len(words),
            'num_sentences': len(sentences),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'num_unique_words': len(set(words)),
            'vocabulary_richness': len(set(words)) / len(words) if words else 0
        }
        return stats
    
    @staticmethod
    def get_top_terms(texts: List[str], n: int = 10) -> List[str]:
        """Get most frequent terms across documents"""
        word_freq = defaultdict(int)
        for text in texts:
            for word in text.split():
                if len(word) > 3:  # Skip short words
                    word_freq[word] += 1
        
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:n]]


class TextClassificationEngine:
    """Main classification engine"""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.analyzer = DocumentAnalyzer()
        self.vectorizer = None
        self.classifier = None
        self.categories = []
        self.model_type = 'nb'  # Default to Naive Bayes
        
    def load_dataset(self, file_path: str = 'data/documents.json') -> Tuple[List[str], List[str]]:
        """Load training dataset"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = []
        labels = []
        
        for category, docs in data.items():
            for doc in docs:
                documents.append(doc['text'])
                labels.append(category)
        
        self.categories = sorted(list(set(labels)))
        print(f"📚 Loaded {len(documents)} documents across {len(self.categories)} categories")
        return documents, labels
    
    def prepare_data(self, texts: List[str], labels: List[str] = None, training: bool = True):
        """Prepare data for training or prediction"""
        # Preprocess texts
        processed_texts = [self.preprocessor.process(text) for text in texts]
        
        if training:
            # Create and fit vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.9
            )
            X = self.vectorizer.fit_transform(processed_texts)
        else:
            # Transform using existing vectorizer
            X = self.vectorizer.transform(processed_texts)
        
        return X, labels
    
    def train(self, X, y, model_type: str = 'nb'):
        """Train classification model"""
        self.model_type = model_type
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Select and train model
        if model_type == 'nb':
            self.classifier = MultinomialNB(alpha=0.1)
        elif model_type == 'lr':
            self.classifier = LogisticRegression(max_iter=500, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )
        
        print(f"\n📊 Model Performance ({model_type.upper()})")
        print(f"   Accuracy: {accuracy:.2%}")
        print(f"   Precision: {precision:.2%}")
        print(f"   Recall: {recall:.2%}")
        print(f"   F1-Score: {f1:.2%}")
        
        # Cross-validation
        cv_results = cross_validate(
            self.classifier, X, y, cv=5,
            scoring=['accuracy', 'precision_weighted', 'recall_weighted']
        )
        
        print(f"\n🔄 Cross-Validation Results:")
        print(f"   Accuracy: {cv_results['test_accuracy'].mean():.2%} "
              f"(± {cv_results['test_accuracy'].std():.2%})")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def classify(self, text: str) -> Dict:
        """Classify a single document"""
        # Prepare text
        X, _ = self.prepare_data([text], training=False)
        
        # Get prediction
        prediction = self.classifier.predict(X)[0]
        
        # Get probabilities
        if hasattr(self.classifier, 'predict_proba'):
            probabilities = self.classifier.predict_proba(X)[0]
            prob_dict = {cat: float(prob) for cat, prob in zip(self.categories, probabilities)}
        else:
            prob_dict = {prediction: 1.0}
        
        # Get text statistics
        stats = self.analyzer.extract_statistics(text)
        
        return {
            'category': prediction,
            'confidence': max(prob_dict.values()),
            'probabilities': prob_dict,
            'statistics': stats
        }
    
    def batch_classify(self, texts: List[str]) -> List[Dict]:
        """Classify multiple documents"""
        results = []
        for text in texts:
            result = self.classify(text)
            results.append(result)
        return results
    
    def save(self, directory: str = 'models/'):
        """Save model and vectorizer"""
        path = Path(directory)
        path.mkdir(exist_ok=True)
        
        # Save vectorizer
        with open(path / 'text_vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        # Save classifier
        with open(path / f'text_classifier_{self.model_type}.pkl', 'wb') as f:
            pickle.dump(self.classifier, f)
        
        # Save categories
        with open(path / 'categories.json', 'w') as f:
            json.dump(self.categories, f)
        
        print(f"✅ Model saved to {directory}")
    
    def load(self, directory: str = 'models/', model_type: str = 'nb'):
        """Load saved model"""
        path = Path(directory)
        self.model_type = model_type
        
        # Load vectorizer
        with open(path / 'text_vectorizer.pkl', 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        # Load classifier
        with open(path / f'text_classifier_{model_type}.pkl', 'rb') as f:
            self.classifier = pickle.load(f)
        
        # Load categories
        with open(path / 'categories.json', 'r') as f:
            self.categories = json.load(f)
        
        print(f"✅ Model loaded from {directory}")


def demonstrate_system():
    """Demonstrate the classification system"""
    print("\n" + "="*50)
    print("TEXT CLASSIFICATION ENGINE")
    print("Author: Abishek Bimali")
    print("="*50)
    
    # Initialize engine
    engine = TextClassificationEngine()
    
    # Load and prepare data
    print("\n1️⃣ Loading Dataset...")
    texts, labels = engine.load_dataset()
    
    print("\n2️⃣ Preparing Features...")
    X, y = engine.prepare_data(texts, labels, training=True)
    print(f"   Feature matrix shape: {X.shape}")
    
    # Train models
    print("\n3️⃣ Training Models...")
    
    # Train Naive Bayes
    nb_metrics = engine.train(X, y, model_type='nb')
    engine.save()
    
    # Train Logistic Regression
    lr_engine = TextClassificationEngine()
    lr_engine.categories = engine.categories
    lr_engine.vectorizer = engine.vectorizer
    lr_metrics = lr_engine.train(X, y, model_type='lr')
    lr_engine.save()
    
    # Test classification
    print("\n4️⃣ Testing Classification...")
    
    test_texts = [
        "The government passed new legislation on healthcare reform.",
        "Stock prices surged following positive earnings reports.",
        "Clinical trials show promising results for new cancer treatment.",
        "Election campaign focuses on economic policies.",
        "Tech companies announce major acquisitions.",
        "Researchers discover link between diet and heart disease."
    ]
    
    print("\n📝 Sample Classifications:")
    for text in test_texts:
        result = engine.classify(text)
        print(f"\nText: '{text[:60]}...'" if len(text) > 60 else f"\nText: '{text}'")
        print(f"→ Category: {result['category'].upper()}")
        print(f"→ Confidence: {result['confidence']:.2%}")
    
    # Analyze dataset
    print("\n5️⃣ Dataset Analysis...")
    analyzer = DocumentAnalyzer()
    processed_texts = [engine.preprocessor.process(text) for text in texts[:50]]
    top_terms = analyzer.get_top_terms(processed_texts, n=15)
    
    print("\n📊 Top Terms in Dataset:")
    for i, term in enumerate(top_terms, 1):
        print(f"   {i:2}. {term}")
    
    print("\n✨ System demonstration complete!")


if __name__ == "__main__":
    demonstrate_system()