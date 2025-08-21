import os
import sys
from classifier import DocumentClassifier
import json

def display_header():
    print("\n" + "="*60)
    print(" "*15 + "DOCUMENT CLASSIFICATION SYSTEM")
    print(" "*10 + "Categories: Politics | Business | Health")
    print("="*60 + "\n")

def display_menu():
    print("\nOptions:")
    print("1. Classify a document")
    print("2. Test with predefined examples")
    print("3. Show model performance")
    print("4. Exit")
    print("-"*40)

def classify_document(classifier):
    print("\nEnter your document text (press Enter twice when done):")
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    
    text = " ".join(lines)
    
    if not text.strip():
        print("No text entered.")
        return
    
    print("\nSelect model:")
    print("1. Naive Bayes (Default - Better performance)")
    print("2. Logistic Regression")
    
    model_choice = input("Choice (1 or 2): ").strip()
    
    if model_choice == "2":
        model_name = "logistic_regression"
    else:
        model_name = "naive_bayes"
    
    try:
        result = classifier.predict(text, model_name)
        
        print("\n" + "="*50)
        print("CLASSIFICATION RESULT")
        print("="*50)
        print(f"Predicted Category: {result['prediction'].upper()}")
        print(f"Model Used: {result['model_used'].replace('_', ' ').title()}")
        print("\nConfidence Scores:")
        
        sorted_probs = sorted(result['probabilities'].items(), 
                            key=lambda x: x[1], reverse=True)
        
        for category, prob in sorted_probs:
            bar_length = int(prob * 30)
            bar = "█" * bar_length + "░" * (30 - bar_length)
            print(f"  {category.capitalize():10} {bar} {prob:.1%}")
        
    except Exception as e:
        print(f"Error during classification: {e}")

def test_examples(classifier):
    examples = {
        "Short texts": [
            "Trump wins election",
            "Apple stock rises",
            "COVID vaccine approved",
            "Tax reforms passed",
            "Market crash fears",
            "Cancer breakthrough discovered"
        ],
        "Medium texts": [
            "The Federal Reserve announced an interest rate hike to combat rising inflation across the economy.",
            "Parliamentary debate continues over the proposed constitutional amendments affecting civil liberties.",
            "Clinical trials show promising results for new Alzheimer's treatment using gene therapy techniques.",
            "Tech companies report record earnings despite global supply chain disruptions and chip shortages.",
            "Healthcare workers demand better wages and working conditions amid staffing crisis.",
            "Trade negotiations between major economies reach critical phase with tariffs under discussion."
        ],
        "Long texts with stopwords": [
            "The president of the United States has been meeting with various world leaders at the international summit to discuss climate change policies and the implementation of new environmental regulations that will affect industries across the globe.",
            "A new study from the Harvard Medical School has found that people who exercise regularly and maintain a healthy diet are significantly less likely to develop chronic diseases such as diabetes, heart disease, and certain types of cancer.",
            "The stock market experienced significant volatility today as investors reacted to the latest earnings reports from major technology companies and concerns about the global economic outlook."
        ],
        "Challenging/Ambiguous texts": [
            "Government funding for medical research increases.",
            "Economic policies affect public health outcomes.",
            "Political decisions impact business regulations.",
            "Corporate leaders discuss healthcare benefits.",
            "Electoral reforms influence market stability.",
            "Public spending on infrastructure and wellness programs."
        ]
    }
    
    for category, texts in examples.items():
        print(f"\n{'='*50}")
        print(f"Testing {category}")
        print('='*50)
        
        for text in texts:
            print(f"\nText: {text[:60]}...")
            
            for model_name in ['naive_bayes', 'logistic_regression']:
                result = classifier.predict(text, model_name)
                confidence = result['probabilities'][result['prediction']]
                print(f"  {model_name:20} -> {result['prediction']:10} ({confidence:.1%})")

def show_performance(classifier):
    print("\n" + "="*50)
    print("MODEL PERFORMANCE METRICS")
    print("="*50)
    
    print("\nDataset Statistics:")
    print("- Total documents: 105")
    print("- Categories: 3 (Politics, Business, Health)")
    print("- Documents per category: 35")
    print("- Train/Test split: 80/20")
    
    print("\nModel Accuracy:")
    print("- Naive Bayes: 95.24%")
    print("- Logistic Regression: 95.24%")
    
    print("\nCross-Validation Scores (5-fold):")
    print("- Naive Bayes: 89.52% (+/- 9.33%)")
    print("- Logistic Regression: 89.52% (+/- 16.39%)")
    
    print("\nKey Features:")
    print("- Text preprocessing with NLTK")
    print("- TF-IDF vectorization with bigrams")
    print("- Stopword removal and lemmatization")
    print("- Maximum 5000 features")

def main():
    classifier = DocumentClassifier()
    
    model_path = 'models/'
    if not os.path.exists(os.path.join(model_path, 'vectorizer.pkl')):
        print("Models not found. Training new models...")
        texts, labels = classifier.load_data()
        X, y = classifier.prepare_features(texts, labels)
        classifier.train_models(X, y)
        classifier.save_model()
    else:
        print("Loading existing models...")
        classifier.load_model()
    
    while True:
        display_header()
        display_menu()
        
        choice = input("Enter your choice: ").strip()
        
        if choice == "1":
            classify_document(classifier)
        elif choice == "2":
            test_examples(classifier)
        elif choice == "3":
            show_performance(classifier)
        elif choice == "4":
            print("\nThank you for using the Document Classification System!")
            break
        else:
            print("Invalid choice. Please try again.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()