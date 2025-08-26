# #!/usr/bin/env python3
# """
# Demonstration script for the Document Classification System
# Shows classification of various real-world examples using BBC news data
# """

# import pickle
# import json
# from datetime import datetime

# def load_models():
#     """Load the trained models"""
#     with open('models/naive_bayes_model.pkl', 'rb') as f:
#         nb_model = pickle.load(f)
#     with open('models/logistic_regression_model.pkl', 'rb') as f:
#         lr_model = pickle.load(f)
#     with open('models/vectorizer.pkl', 'rb') as f:
#         vectorizer = pickle.load(f)
#     with open('models/label_encoder.pkl', 'rb') as f:
#         label_encoder = pickle.load(f)
    
#     return nb_model, lr_model, vectorizer, label_encoder

# def classify_text(text, models):
#     """Classify a text using both models"""
#     nb_model, lr_model, vectorizer, label_encoder = models
    
#     # Transform the text
#     X = vectorizer.transform([text])
    
#     # Get predictions
#     nb_pred = nb_model.predict(X)[0]
#     nb_proba = nb_model.predict_proba(X)[0]
#     nb_label = label_encoder.inverse_transform([nb_pred])[0]
#     nb_confidence = max(nb_proba) * 100
    
#     lr_pred = lr_model.predict(X)[0]
#     lr_proba = lr_model.predict_proba(X)[0]
#     lr_label = label_encoder.inverse_transform([lr_pred])[0]
#     lr_confidence = max(lr_proba) * 100
    
#     return {
#         'naive_bayes': {'label': nb_label, 'confidence': nb_confidence},
#         'logistic_regression': {'label': lr_label, 'confidence': lr_confidence}
#     }

# def main():
#     print("=" * 70)
#     print("        DOCUMENT CLASSIFICATION SYSTEM - DEMONSTRATION")
#     print("             Using Real BBC News Data Collection")
#     print("=" * 70)
#     print("\nData Source: BBC News - https://www.bbc.com/news")
#     print("Categories: Politics | Business | Health")
#     print("Collection Date: 2025-08-25")
#     print("\nCopyright Notice: All content excerpts are from BBC News")
#     print("Used for educational purposes under fair use provisions")
#     print("=" * 70)
    
#     # Load models
#     print("\nLoading trained models...")
#     models = load_models()
#     print("✓ Models loaded successfully")
    
#     # Test examples from different categories
#     test_examples = [
#         {
#             'category': 'POLITICS',
#             'texts': [
#                 "The Prime Minister announced sweeping reforms to the electoral system during today's parliamentary session.",
#                 "Brexit negotiations continue as UK and EU officials meet to discuss trade agreements.",
#                 "Opposition parties unite to challenge government's controversial immigration bill."
#             ]
#         },
#         {
#             'category': 'BUSINESS',
#             'texts': [
#                 "Tech stocks surge as major companies report better than expected quarterly earnings.",
#                 "The Bank of England raises interest rates to combat rising inflation across the UK economy.",
#                 "Small businesses struggle with supply chain disruptions and rising operational costs."
#             ]
#         },
#         {
#             'category': 'HEALTH',
#             'texts': [
#                 "NHS waiting lists reach record highs as hospitals face staff shortages and capacity issues.",
#                 "Breakthrough cancer treatment shows promising results in clinical trials at UK research centers.",
#                 "Mental health services expand digital therapy options to meet unprecedented demand."
#             ]
#         },
#         {
#             'category': 'CHALLENGING/AMBIGUOUS',
#             'texts': [
#                 "Government announces billion pound investment in NHS infrastructure development.",
#                 "Pharmaceutical companies lobby parliament for regulatory changes affecting drug pricing.",
#                 "Economic downturn forces cuts to public health programs across the country."
#             ]
#         }
#     ]
    
#     # Classify each example
#     for example_set in test_examples:
#         print(f"\n{'=' * 70}")
#         print(f"Testing {example_set['category']} Examples")
#         print('=' * 70)
        
#         for i, text in enumerate(example_set['texts'], 1):
#             print(f"\nExample {i}:")
#             print(f"Text: \"{text[:80]}...\"" if len(text) > 80 else f"Text: \"{text}\"")
            
#             results = classify_text(text, models)
            
#             print(f"\nClassification Results:")
#             print(f"  • Naive Bayes:        {results['naive_bayes']['label'].upper()} " +
#                   f"(confidence: {results['naive_bayes']['confidence']:.1f}%)")
#             print(f"  • Logistic Regression: {results['logistic_regression']['label'].upper()} " +
#                   f"(confidence: {results['logistic_regression']['confidence']:.1f}%)")
            
#             # Check if models agree
#             if results['naive_bayes']['label'] == results['logistic_regression']['label']:
#                 print(f"  ✓ Both models agree: {results['naive_bayes']['label'].upper()}")
#             else:
#                 print(f"  ⚠ Models disagree - NB: {results['naive_bayes']['label']}, " +
#                       f"LR: {results['logistic_regression']['label']}")
    
#     # Interactive classification
#     print(f"\n{'=' * 70}")
#     print("INTERACTIVE CLASSIFICATION")
#     print('=' * 70)
#     print("\nYou can now enter your own text for classification.")
#     print("Type 'quit' to exit.\n")
    
#     while True:
#         user_text = input("\nEnter text to classify (or 'quit'): ").strip()
        
#         if user_text.lower() == 'quit':
#             break
        
#         if not user_text:
#             print("Please enter some text to classify.")
#             continue
        
#         results = classify_text(user_text, models)
        
#         print(f"\nClassification Results:")
#         print(f"  • Naive Bayes:        {results['naive_bayes']['label'].upper()} " +
#               f"(confidence: {results['naive_bayes']['confidence']:.1f}%)")
#         print(f"  • Logistic Regression: {results['logistic_regression']['label'].upper()} " +
#               f"(confidence: {results['logistic_regression']['confidence']:.1f}%)")
    
#     print("\n" + "=" * 70)
#     print("Thank you for using the Document Classification System!")
#     print("=" * 70)

# if __name__ == "__main__":
#     main()