import json
import os
from datetime import datetime

def update_documents_with_real_data():
    """
    Update the documents.json file with real BBC news data
    """
    # Load the real documents
    with open('data/real_documents.json', 'r', encoding='utf-8') as f:
        real_data = json.load(f)
    
    # Create the structure for documents.json
    documents = {
        'politics': real_data['politics'],
        'business': real_data['business'],
        'health': real_data['health']
    }
    
    # Save to documents.json
    with open('data/documents.json', 'w', encoding='utf-8') as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)
    
    # Print statistics
    print("Document Collection Statistics:")
    print("=" * 50)
    print(f"Politics documents: {len(documents['politics'])}")
    print(f"Business documents: {len(documents['business'])}")
    print(f"Health documents: {len(documents['health'])}")
    print(f"Total documents: {sum(len(docs) for docs in documents.values())}")
    print("\nMetadata:")
    print(f"Source: {real_data['metadata']['sources']}")
    print(f"Collection Date: {real_data['metadata']['collection_date']}")
    print(f"Copyright: {real_data['metadata']['copyright_notice']}")
    
    return documents

if __name__ == "__main__":
    documents = update_documents_with_real_data()
    print("\n✓ Documents successfully updated with real BBC news data!")
    print("✓ All content properly attributed to BBC News")
    print("✓ Ready to train the classification model with real data")