#!/usr/bin/env python3
"""
Document Classification Application
Author: Abishek Bimali
Date: 2025
"""

import os
import sys
from pathlib import Path
from typing import List, Dict
import json
from text_classifier_engine import TextClassificationEngine


class ClassificationApp:
    """Simple interactive classification application"""
    
    def __init__(self):
        self.engine = TextClassificationEngine()
        self.model_loaded = False
        
    def display_banner(self):
        """Display application banner"""
        print("\n" + "╔" + "═"*48 + "╗")
        print("║" + " "*15 + "DOCUMENT CLASSIFIER" + " "*14 + "║")
        print("║" + " "*48 + "║")
        print("║" + " "*10 + "Categories: Politics | Business | Health" + " "*1 + "║")
        print("║" + " "*17 + "Author: Abishek Bimali" + " "*9 + "║")
        print("╚" + "═"*48 + "╝")
    
    def setup_models(self):
        """Initialize or train models"""
        model_dir = Path('models/')
        model_file = model_dir / 'text_classifier_nb.pkl'
        
        if model_file.exists():
            print("\n📂 Loading existing models...")
            try:
                self.engine.load()
                self.model_loaded = True
                print("✅ Models loaded successfully!")
            except Exception as e:
                print(f"❌ Error loading models: {e}")
                self.train_new_models()
        else:
            print("\n⚠️  No trained models found.")
            self.train_new_models()
    
    def train_new_models(self):
        """Train new classification models"""
        print("\n🔧 Training new models...")
        
        try:
            # Load dataset
            texts, labels = self.engine.load_dataset()
            
            # Prepare features
            X, y = self.engine.prepare_data(texts, labels, training=True)
            
            # Train model
            self.engine.train(X, y, model_type='nb')
            
            # Save model
            self.engine.save()
            
            self.model_loaded = True
            print("✅ Training completed!")
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            sys.exit(1)
    
    def classify_text(self):
        """Classify user-provided text"""
        print("\n" + "-"*50)
        print("CLASSIFY DOCUMENT")
        print("-"*50)
        print("Enter your text (type 'DONE' on a new line when finished):")
        print()
        
        lines = []
        while True:
            line = input()
            if line.strip().upper() == 'DONE':
                break
            lines.append(line)
        
        text = ' '.join(lines).strip()
        
        if not text:
            print("❌ No text provided!")
            return
        
        # Classify
        print("\n⏳ Analyzing text...")
        result = self.engine.classify(text)
        
        # Display results
        print("\n" + "="*50)
        print("CLASSIFICATION RESULTS")
        print("="*50)
        
        print(f"\n📌 Category: {result['category'].upper()}")
        print(f"📊 Confidence: {result['confidence']:.1%}")
        
        print("\n📈 Probability Distribution:")
        for category in sorted(result['probabilities'].keys()):
            prob = result['probabilities'][category]
            bar_length = int(prob * 30)
            bar = "█" * bar_length + "░" * (30 - bar_length)
            print(f"  {category:10} {bar} {prob:.1%}")
        
        print("\n📝 Text Statistics:")
        stats = result['statistics']
        print(f"  • Words: {stats['num_words']}")
        print(f"  • Sentences: {stats['num_sentences']}")
        print(f"  • Vocabulary Richness: {stats['vocabulary_richness']:.2%}")
    
    def test_examples(self):
        """Test with predefined examples"""
        print("\n" + "-"*50)
        print("TEST EXAMPLES")
        print("-"*50)
        
        examples = {
            "Politics": [
                "The president announced new foreign policy initiatives.",
                "Congress debates infrastructure spending bill.",
                "Election polls show tight race in swing states."
            ],
            "Business": [
                "Tech stocks rally on strong quarterly earnings.",
                "Federal Reserve discusses interest rate changes.",
                "Startup secures major venture capital funding."
            ],
            "Health": [
                "New study reveals benefits of Mediterranean diet.",
                "Vaccine development progresses for tropical diseases.",
                "Mental health awareness campaign launches nationwide."
            ]
        }
        
        print("\nClassifying example texts...\n")
        
        for expected_category, texts in examples.items():
            print(f"📁 {expected_category} Examples:")
            print("-" * 40)
            
            for text in texts:
                result = self.engine.classify(text)
                
                # Check if prediction matches expected
                symbol = "✅" if result['category'].lower() == expected_category.lower() else "❌"
                
                print(f"{symbol} Text: '{text[:50]}...'" if len(text) > 50 else f"{symbol} Text: '{text}'")
                print(f"   → Predicted: {result['category'].upper()} ({result['confidence']:.1%})")
            print()
    
      
    def display_menu(self):
        """Display main menu"""
        print("\n" + "="*50)
        print("MAIN MENU")
        print("="*50)
        print("1. 📝 Classify a document")
        print("2. 🧪 Test with examples")
        print("3. 🔄 Retrain models")
        print("5. 🚪 Exit")
        print("-"*50)
    
    def run(self):
        """Main application loop"""
        self.display_banner()
        self.setup_models()
        
        while True:
            self.display_menu()
            
            choice = input("\nSelect option (1-5): ").strip()
            
            if choice == '1':
                self.classify_text()
            elif choice == '2':
                self.test_examples()
            elif choice == '3':
                self.train_new_models()
            elif choice == '5':
                print("\n👋 Thank you for using Document Classifier!")
                print("   Created by Abishek Bimali\n")
                break
            else:
                print("❌ Invalid choice! Please select 1-5")
            
            if choice in ['1', '2', '3']:
                input("\nPress Enter to continue...")


def main():
    """Entry point"""
    app = ClassificationApp()
    try:
        app.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Application interrupted by user")
        print("👋 Goodbye!\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()