#!/usr/bin/env python3
import json
import re
import os
from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import string
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Coventry University Publications Search Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Publication(BaseModel):
    title: str
    authors: List[str]
    link: str
    published_date: str
    abstract: str
    author_profiles: Optional[Dict[str, str]] = {}
    score: Optional[float] = None

class SearchResult(BaseModel):
    query: str
    total_results: int
    results: List[Publication]
    search_time: float

class SearchEngine:
    def __init__(self, data_path: str = "./data"):
        self.data_path = Path(data_path)
        self.publications = []
        self.dept_members = []
        self.author_index = {}  # Map author names to publication indices
        self.year_index = {}    # Map years to publication indices
        self.keyword_index = {} # Inverted index for keyword search
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,  # Increased for better coverage
            stop_words='english',
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.90,
            sublinear_tf=True,  # Use log normalization
            use_idf=True
        )
        self.tfidf_matrix = None
        self.load_data()
        self.build_index()
        self.build_inverted_indices()
    
    def load_data(self):
        """Load all data from JSON files"""
        try:
            publications_path = self.data_path / "publications.json"
            if publications_path.exists():
                with open(publications_path, 'r', encoding='utf-8') as f:
                    self.publications = json.load(f)
                logger.info(f"Loaded {len(self.publications)} publications")
            else:
                publications_full_path = self.data_path / "publications_full.json"
                if publications_full_path.exists():
                    with open(publications_full_path, 'r', encoding='utf-8') as f:
                        self.publications = json.load(f)
                    logger.info(f"Loaded {len(self.publications)} publications from full file")
            
            # Author profiles are now stored directly in each publication record
            
            dept_members_path = self.data_path / "dept_members.json"
            if dept_members_path.exists():
                with open(dept_members_path, 'r', encoding='utf-8') as f:
                    self.dept_members = json.load(f)
                logger.info(f"Loaded {len(self.dept_members)} department members")
                
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for indexing and searching"""
        if not text:
            return ""
        text = text.lower()
        # Keep some punctuation that might be meaningful (like hyphens in compound words)
        text = re.sub(r'[^\w\s\-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text for inverted index"""
        if not text:
            return []
        # Remove punctuation and split into words
        words = self.preprocess_text(text).split()
        # Filter out very short words and common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were'}
        keywords = [w for w in words if len(w) > 2 and w not in stop_words]
        return keywords
    
    def build_index(self):
        """Build TF-IDF index for publications"""
        if not self.publications:
            logger.warning("No publications to index")
            return
        
        documents = []
        for pub in self.publications:
            # Give more weight to title and authors in the document representation
            title = pub.get('title', '')
            authors = " ".join(pub.get('authors', []))
            abstract = pub.get('abstract', '')
            date = pub.get('published_date', '')
            
            # Duplicate title and authors to give them more weight
            doc_text = " ".join([
                title, title,  # Title appears twice for more weight
                authors, authors,  # Authors appear twice
                abstract,
                date
            ])
            documents.append(self.preprocess_text(doc_text))
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
        logger.info(f"Built TF-IDF index with shape {self.tfidf_matrix.shape}")
    
    def build_inverted_indices(self):
        """Build inverted indices for faster filtering and keyword search"""
        if not self.publications:
            return
        
        self.author_index = {}
        self.year_index = {}
        self.keyword_index = {}
        
        for idx, pub in enumerate(self.publications):
            # Build author index
            for author in pub.get('authors', []):
                author_lower = author.lower()
                if author_lower not in self.author_index:
                    self.author_index[author_lower] = []
                self.author_index[author_lower].append(idx)
            
            # Build year index
            date_str = pub.get('published_date', '')
            year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
            if year_match:
                year = year_match.group()
                if year not in self.year_index:
                    self.year_index[year] = []
                self.year_index[year].append(idx)
            
            # Build keyword index from title and abstract
            text = pub.get('title', '') + ' ' + pub.get('abstract', '')
            keywords = self.extract_keywords(text)
            for keyword in keywords:
                if keyword not in self.keyword_index:
                    self.keyword_index[keyword] = set()
                self.keyword_index[keyword].add(idx)
        
        # Convert sets to lists for JSON serialization if needed
        for keyword in self.keyword_index:
            self.keyword_index[keyword] = list(self.keyword_index[keyword])
        
        logger.info(f"Built inverted indices: {len(self.author_index)} authors, {len(self.year_index)} years, {len(self.keyword_index)} keywords")
    
    def search(self, query: str, max_results: int = 20, 
               author_filter: Optional[str] = None,
               year_filter: Optional[str] = None) -> List[Dict]:
        """
        Enhanced search using TF-IDF, keyword matching, and filtering
        """
        if not query:
            return []
        
        # Process query for TF-IDF
        processed_query = self.preprocess_text(query)
        
        # Duplicate important query terms for emphasis
        query_words = processed_query.split()
        enhanced_query = processed_query
        if len(query_words) <= 3:  # For short queries, duplicate all terms
            enhanced_query = " ".join([processed_query, processed_query])
        
        query_vector = self.tfidf_vectorizer.transform([enhanced_query])
        
        # Calculate TF-IDF similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get candidate indices based on filters
        candidate_indices = set(range(len(self.publications)))
        
        # Apply author filter using index
        if author_filter:
            author_lower = author_filter.lower()
            filtered_indices = set()
            for author_key in self.author_index:
                if author_lower in author_key:
                    filtered_indices.update(self.author_index[author_key])
            if filtered_indices:
                candidate_indices &= filtered_indices
        
        # Apply year filter using index
        if year_filter:
            if year_filter in self.year_index:
                candidate_indices &= set(self.year_index[year_filter])
            else:
                candidate_indices = set()  # No publications for this year
        
        # Boost scores for exact keyword matches
        query_keywords = self.extract_keywords(query)
        keyword_boost = np.zeros(len(self.publications))
        
        for keyword in query_keywords:
            if keyword in self.keyword_index:
                for idx in self.keyword_index[keyword]:
                    keyword_boost[idx] += 0.1  # Boost score for keyword match
        
        # Combine TF-IDF scores with keyword boosts
        combined_scores = similarities + keyword_boost
        
        # Filter and rank results
        scored_candidates = []
        for idx in candidate_indices:
            score = float(combined_scores[idx])
            if score > 0.01:  # Lower threshold for including results
                scored_candidates.append((idx, score))
        
        # Sort by score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Build results
        results = []
        for idx, score in scored_candidates[:max_results]:
            pub = self.publications[idx].copy()
            pub['score'] = score
            
            # Ensure author_profiles field exists (it should already be in the data)
            if 'author_profiles' not in pub:
                pub['author_profiles'] = {}
            
            results.append(pub)
        
        # If no results found with TF-IDF, try basic keyword matching
        if not results and query_keywords:
            keyword_matches = set()
            for keyword in query_keywords:
                if keyword in self.keyword_index:
                    keyword_matches.update(self.keyword_index[keyword])
            
            # Apply filters to keyword matches
            keyword_matches &= candidate_indices
            
            for idx in list(keyword_matches)[:max_results]:
                pub = self.publications[idx].copy()
                pub['score'] = 0.5  # Default score for keyword-only matches
                results.append(pub)
        
        return results
    
    def get_statistics(self) -> Dict:
        """Get statistics about the indexed data"""
        total_authors = set()
        years = []
        
        for pub in self.publications:
            for author in pub.get('authors', []):
                total_authors.add(author)
            
            date_str = pub.get('published_date', '')
            year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
            if year_match:
                years.append(year_match.group())
        
        return {
            'total_publications': len(self.publications),
            'total_unique_authors': len(total_authors),
            'total_dept_members': len(self.dept_members),
            'years_range': f"{min(years) if years else 'N/A'} - {max(years) if years else 'N/A'}",
            'indexed_at': datetime.now().isoformat()
        }

search_engine = SearchEngine()

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Coventry University Publications Search Engine API",
        "endpoints": {
            "/search": "Search publications",
            "/stats": "Get statistics",
            "/publications": "Get all publications",
            "/authors": "Get all authors"
        }
    }

@app.get("/search", response_model=SearchResult)
async def search_publications(
    q: str = Query(..., description="Search query"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of results"),
    author: Optional[str] = Query(None, description="Filter by author name"),
    year: Optional[str] = Query(None, description="Filter by publication year")
):
    """Search for publications"""
    start_time = datetime.now()
    
    if not q:
        raise HTTPException(status_code=400, detail="Query parameter is required")
    
    results = search_engine.search(q, max_results=limit, author_filter=author, year_filter=year)
    
    search_time = (datetime.now() - start_time).total_seconds()
    
    return SearchResult(
        query=q,
        total_results=len(results),
        results=results,
        search_time=search_time
    )

@app.get("/stats")
async def get_statistics():
    """Get statistics about the indexed data"""
    return search_engine.get_statistics()

@app.get("/publications")
async def get_all_publications(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0)
):
    """Get all publications with pagination"""
    end_idx = min(offset + limit, len(search_engine.publications))
    return {
        "total": len(search_engine.publications),
        "limit": limit,
        "offset": offset,
        "publications": search_engine.publications[offset:end_idx]
    }

@app.get("/authors")
async def get_all_authors():
    """Get all unique authors"""
    authors = set()
    for pub in search_engine.publications:
        for author in pub.get('authors', []):
            authors.add(author)
    
    return {
        "total": len(authors),
        "authors": sorted(list(authors))
    }

@app.get("/search/advanced")
async def advanced_search(
    title: Optional[str] = Query(None, description="Search in title only"),
    abstract: Optional[str] = Query(None, description="Search in abstract only"),
    author: Optional[str] = Query(None, description="Search by author name"),
    year_from: Optional[str] = Query(None, description="Publications from year"),
    year_to: Optional[str] = Query(None, description="Publications until year"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of results")
):
    """Advanced search with field-specific queries"""
    results = []
    
    for idx, pub in enumerate(search_engine.publications):
        # Check title match
        if title and title.lower() not in pub.get('title', '').lower():
            continue
        
        # Check abstract match
        if abstract and abstract.lower() not in pub.get('abstract', '').lower():
            continue
        
        # Check author match
        if author:
            author_match = any(author.lower() in a.lower() for a in pub.get('authors', []))
            if not author_match:
                continue
        
        # Check year range
        date_str = pub.get('published_date', '')
        year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
        if year_match:
            pub_year = int(year_match.group())
            if year_from and pub_year < int(year_from):
                continue
            if year_to and pub_year > int(year_to):
                continue
        
        results.append(pub)
        
        if len(results) >= limit:
            break
    
    return {
        "total": len(results),
        "results": results
    }

@app.get("/trending")
async def get_trending_topics():
    """Get trending research topics based on recent publications"""
    from datetime import datetime
    current_year = datetime.now().year
    recent_years = [str(current_year - i) for i in range(3)]  # Last 3 years
    
    # Count keywords in recent publications
    keyword_counts = Counter()
    
    for pub in search_engine.publications:
        date_str = pub.get('published_date', '')
        if any(year in date_str for year in recent_years):
            text = pub.get('title', '') + ' ' + pub.get('abstract', '')
            keywords = search_engine.extract_keywords(text)
            keyword_counts.update(keywords)
    
    # Get top 20 trending keywords
    trending = keyword_counts.most_common(20)
    
    return {
        "trending_topics": [{"keyword": k, "count": c} for k, c in trending],
        "analysis_period": f"{recent_years[-1]}-{recent_years[0]}"
    }

@app.get("/reindex")
async def reindex_data():
    """Reload data and rebuild index"""
    try:
        search_engine.load_data()
        search_engine.build_index()
        search_engine.build_inverted_indices()
        return {"message": "Data reindexed successfully", "stats": search_engine.get_statistics()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reindexing: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)