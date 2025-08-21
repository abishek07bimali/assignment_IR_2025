# Document Classification System

## Overview
This project implements a document classification system that categorizes text documents into three categories: **Politics**, **Business**, and **Health**. The system uses machine learning algorithms (Naive Bayes and Logistic Regression) to automatically classify new documents based on their content.

## Features
- **Multi-class text classification** for Politics, Business, and Health categories
- **Two classification models**: Naive Bayes and Logistic Regression
- **Text preprocessing** with NLTK (tokenization, lemmatization, stopword removal)
- **TF-IDF vectorization** with bigram support
- **Interactive command-line interface** for easy document classification
- **Comprehensive testing** with various text lengths and complexities
- **Model persistence** for quick loading without retraining

## Project Structure
```
document_classifier/
├── data/
│   └── documents.json          # Training dataset (105 documents)
├── models/
│   ├── vectorizer.pkl          # TF-IDF vectorizer
│   ├── label_encoder.pkl       # Label encoder
│   ├── naive_bayes_model.pkl   # Trained NB model
│   └── logistic_regression_model.pkl  # Trained LR model
├── data_collector.py           # Document collection script
├── classifier.py               # Main classification engine
├── app.py                      # Interactive CLI application
├── test_system.py              # Comprehensive testing script
├── test_results.json           # Test results output
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation

1. Clone or download the project to your local machine
2. Navigate to the project directory:
```bash
cd document_classifier
```
3. Install required dependencies:
```bash
pip3 install -r requirements.txt
```

## Usage

### Running the Interactive Application
```bash
python3 app.py
```

The application provides the following options:
1. **Classify a document** - Enter any text to get its classification
2. **Test with predefined examples** - Run tests on various text types
3. **Show model performance** - View accuracy metrics and statistics
4. **Exit** - Close the application

### Training New Models
If you want to retrain the models with new data:
```bash
python3 classifier.py
```

### Running Comprehensive Tests
To evaluate the system with various input types:
```bash
python3 test_system.py
```

## Dataset

The system is trained on 105 documents (35 per category):
- **Politics**: Government policies, elections, diplomatic relations, legislation
- **Business**: Market trends, corporate news, financial reports, economic indicators
- **Health**: Medical research, public health, diseases, treatments, wellness

### Data Sources and Citations
The training documents are sample texts created to represent typical content from each category, inspired by common news article structures and topics found in:
- General news website formats (BBC News, CNN, Reuters style)
- Public domain news content structures
- Academic and research paper abstracts

**Note**: All training data consists of synthetic examples created for educational purposes. No copyrighted content from actual news sources was directly copied.

## Model Performance

### Accuracy Metrics
- **Naive Bayes**: 95.24% accuracy (test set)
- **Logistic Regression**: 95.24% accuracy (test set)
- **Cross-validation score**: 89.52% (±9.33% for NB, ±16.39% for LR)

### Comprehensive Test Results
The system was tested with various input types achieving **91.67% overall accuracy**:
- Very short texts (1-3 words): 83% accuracy
- Short texts without stopwords: 100% accuracy
- Medium texts with stopwords: 100% accuracy
- Long detailed texts: 100% accuracy
- Ambiguous/cross-domain texts: 83% accuracy
- Technical/specialized texts: 83% accuracy

## Technical Details

### Text Preprocessing
1. Convert to lowercase
2. Remove special characters and numbers
3. Tokenization using NLTK
4. Remove stopwords (optional)
5. Lemmatization using WordNetLemmatizer

### Feature Extraction
- **TF-IDF Vectorization** with:
  - Maximum 5000 features
  - Unigrams and bigrams (ngram_range=(1,2))
  - Automatic vocabulary learning from training data

### Classification Models
1. **Multinomial Naive Bayes**
   - Alpha smoothing parameter: 0.1
   - Better for shorter texts and quick predictions
   
2. **Logistic Regression**
   - Maximum iterations: 1000
   - L2 regularization (default)
   - Better for well-defined, longer texts

## Testing Robustness

The system has been tested with:
- **Short inputs**: Single words to short phrases
- **Long inputs**: Multi-sentence paragraphs
- **With/without stopwords**: Both preprocessed and raw text
- **Different topics**: Wide range within each category
- **Ambiguous content**: Cross-domain texts that could belong to multiple categories
- **Technical language**: Specialized vocabulary and jargon

## Requirements

- Python 3.7+
- scikit-learn 1.3.2
- pandas 2.1.4
- numpy 1.24.3
- nltk 3.8.1
- matplotlib 3.8.2
- seaborn 0.13.0

## Future Improvements

Potential enhancements could include:
- Adding more categories (Sports, Technology, Entertainment)
- Implementing deep learning models (LSTM, BERT)
- Web scraping for automatic data collection
- REST API for web integration
- Real-time classification of news feeds
- Multi-language support

## License and Usage

This project is created for educational purposes as part of an academic assignment. The code is provided as-is for learning and demonstration purposes.

## Author

Document Classification System - Academic Project
Created for demonstration of text classification techniques using machine learning.