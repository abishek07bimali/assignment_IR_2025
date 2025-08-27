# Document Classification System

**Developer:** Abishek Bimali  


## Overview
A machine learning text classification system using ensemble methods (Naive Bayes, Logistic Regression, SVM) to categorize documents into Politics, Business, or Health categories.

## Structure
```
document_classifier/
├── text_classifier_engine.py    # Core classification engine
├── classification_app.py        # Interactive CLI application
├── data/
│   └── documents.json          # Training dataset
├── models/                     # Saved model files
│   ├── text_classifier_nb.pkl
│   ├── text_vectorizer.pkl
│   └── categories.json
└── requirements.txt
```

## Running Instructions

### Setup
```bash
# Install dependencies
pip install -r requirements.txt
```

### Run Interactive Application
```bash
# Start the classification app
python classification_app.py
```

### Train New Model
```bash
# Train the classifier
python text_classifier_engine.py
```

## Usage Options
1. **Interactive Mode**: Use the CLI menu to classify text or files
2. **Batch Processing**: Process multiple documents at once
3. **API Usage**: Import and use the classifier in your code

```python
from text_classifier_engine import TextClassifier

classifier = TextClassifier()
classifier.load_model('models/')
result = classifier.predict("Your text here")
```