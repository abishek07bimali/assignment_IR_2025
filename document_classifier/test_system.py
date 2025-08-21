from classifier import DocumentClassifier
import json

def comprehensive_test():
    classifier = DocumentClassifier()
    classifier.load_model()
    
    test_cases = {
        "Very Short (1-3 words)": [
            ("Election results announced", "politics"),
            ("Stock crash", "business"),
            ("Vaccine approved", "health"),
            ("President speaks", "politics"),
            ("Market boom", "business"),
            ("Cancer cure", "health")
        ],
        
        "Short without stopwords": [
            ("Parliament debates immigration policy", "politics"),
            ("Amazon expands logistics network", "business"),
            ("Diabetes research breakthrough announced", "health"),
            ("Senator proposes tax reform", "politics"),
            ("Startup raises million funding", "business"),
            ("Mental health awareness campaign", "health")
        ],
        
        "Medium with stopwords": [
            ("The government has been working on a new policy that will affect how citizens vote in the upcoming elections", "politics"),
            ("A major corporation is planning to acquire several smaller companies in order to expand its market share", "business"),
            ("Doctors are recommending that people should get regular checkups to prevent serious health conditions", "health")
        ],
        
        "Long detailed texts": [
            ("The international summit brought together world leaders from over 50 nations to discuss pressing global issues including climate change, economic cooperation, and security concerns. The discussions focused particularly on establishing new frameworks for diplomatic engagement and peaceful resolution of conflicts. Several bilateral meetings were held on the sidelines where trade agreements and defense partnerships were negotiated.", "politics"),
            
            ("The quarterly earnings report exceeded analyst expectations with revenue growth of 25% year-over-year, driven primarily by strong performance in the cloud computing division and enterprise software sales. The company's CEO attributed the success to strategic investments in artificial intelligence and machine learning capabilities, which have enhanced product offerings and improved operational efficiency. Looking forward, management provided optimistic guidance for the next fiscal year.", "business"),
            
            ("A comprehensive longitudinal study involving 10,000 participants over a 15-year period has revealed significant correlations between lifestyle factors and the development of chronic diseases. The research, published in a leading medical journal, demonstrates that individuals who maintain regular physical activity, balanced nutrition, and adequate sleep patterns show a 40% reduced risk of developing cardiovascular disease, type 2 diabetes, and certain forms of cancer.", "health")
        ],
        
        "Ambiguous/Cross-domain": [
            ("Government allocates budget for hospital construction", "health/politics"),
            ("Pharmaceutical company faces regulatory investigation", "business/health"),
            ("Election outcome affects stock market performance", "politics/business"),
            ("Public health emergency declared by officials", "health/politics"),
            ("Insurance industry lobbies for healthcare reform", "business/health"),
            ("Economic downturn impacts mental health services", "business/health")
        ],
        
        "Technical/Specialized": [
            ("Constitutional amendment requires two-thirds majority in both houses of legislature", "politics"),
            ("Leveraged buyout financed through combination of equity and debt instruments", "business"),
            ("Randomized controlled trial demonstrates efficacy of novel therapeutic intervention", "health"),
            ("Geopolitical tensions escalate following territorial disputes in contested region", "politics"),
            ("Quantitative easing measures implemented to stimulate economic growth", "business"),
            ("Epidemiological surveillance systems detect emerging infectious disease patterns", "health")
        ]
    }
    
    print("="*70)
    print(" "*20 + "COMPREHENSIVE SYSTEM TEST")
    print("="*70)
    
    results = {"correct": 0, "total": 0}
    detailed_results = []
    
    for test_type, cases in test_cases.items():
        print(f"\n{'='*60}")
        print(f"Testing: {test_type}")
        print('='*60)
        
        for text, expected in cases:
            nb_result = classifier.predict(text, 'naive_bayes')
            lr_result = classifier.predict(text, 'logistic_regression')
            
            nb_pred = nb_result['prediction']
            lr_pred = lr_result['prediction']
            nb_conf = nb_result['probabilities'][nb_pred]
            lr_conf = lr_result['probabilities'][lr_pred]
            
            if '/' in expected:
                expected_categories = expected.split('/')
                nb_correct = nb_pred in expected_categories
                lr_correct = lr_pred in expected_categories
            else:
                nb_correct = nb_pred == expected
                lr_correct = lr_pred == expected
            
            print(f"\nText: '{text[:80]}...'" if len(text) > 80 else f"\nText: '{text}'")
            print(f"Expected: {expected}")
            print(f"NB: {nb_pred} ({nb_conf:.1%}) {'✓' if nb_correct else '✗'}")
            print(f"LR: {lr_pred} ({lr_conf:.1%}) {'✓' if lr_correct else '✗'}")
            
            results["total"] += 2
            if nb_correct:
                results["correct"] += 1
            if lr_correct:
                results["correct"] += 1
            
            detailed_results.append({
                "test_type": test_type,
                "text": text[:100],
                "expected": expected,
                "nb_prediction": nb_pred,
                "nb_confidence": nb_conf,
                "nb_correct": nb_correct,
                "lr_prediction": lr_pred,
                "lr_confidence": lr_conf,
                "lr_correct": lr_correct
            })
    
    print("\n" + "="*70)
    print("OVERALL RESULTS")
    print("="*70)
    print(f"Total predictions: {results['total']}")
    print(f"Correct predictions: {results['correct']}")
    print(f"Overall accuracy: {results['correct']/results['total']*100:.2f}%")
    
    with open('test_results.json', 'w') as f:
        json.dump(detailed_results, f, indent=2)
    print("\nDetailed results saved to test_results.json")

if __name__ == "__main__":
    comprehensive_test()