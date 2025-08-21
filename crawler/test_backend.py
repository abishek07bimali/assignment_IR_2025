#!/usr/bin/env python3
import requests

# Test the backend API
base_url = "http://localhost:8000"

# First, let's check if the backend is running
try:
    response = requests.get(f"{base_url}/stats")
    if response.status_code == 200:
        print("✅ Backend is running")
        stats = response.json()
        print(f"Total publications: {stats.get('total_publications', 0)}")
    else:
        print(f"❌ Backend returned status {response.status_code}")
except requests.exceptions.ConnectionError:
    print("❌ Backend is not running. Start it with: python3 search_engine_backend.py")
    exit(1)

# Test search functionality
test_query = "finance"
print(f"\n🔍 Testing search for: '{test_query}'")

response = requests.get(f"{base_url}/search", params={"q": test_query, "limit": 2})
if response.status_code == 200:
    data = response.json()
    print(f"Found {data['total_results']} results")
    
    for i, pub in enumerate(data['results'], 1):
        print(f"\n📄 Result {i}:")
        print(f"  Title: {pub['title'][:80]}...")
        print(f"  Authors: {', '.join(pub['authors'])}")
        
        # Check if author_profiles exists and has data
        if 'author_profiles' in pub:
            if pub['author_profiles']:
                print(f"  ✅ Author profiles available:")
                for author, profile_url in pub['author_profiles'].items():
                    print(f"    - {author}: {profile_url[:60]}...")
            else:
                print(f"  ℹ️ No author profile links available")
        else:
            print(f"  ⚠️ author_profiles field missing")
        
        print(f"  Score: {pub.get('score', 0):.3f}")
else:
    print(f"❌ Search failed with status {response.status_code}")