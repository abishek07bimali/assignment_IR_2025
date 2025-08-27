#!/usr/bin/env python3
"""
Academic Research Search Engine
Author: Abishek Bimali
Date: 2025
Description: Advanced search engine for academic publications with intelligent ranking
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass, field
from collections import defaultdict
import math

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI Application
app = FastAPI(
    title="Academic Research Search Engine",
    description="Intelligent search system for academic publications",
    version="2.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@dataclass
class ResearchPaper:
    """Data class for research papers"""
    title: str
    authors: List[str]
    link: str
    published_date: str
    abstract: str
    author_profiles: Dict[str, str] = field(default_factory=dict)
    keywords: List[str] = field(default_factory=list)
    citations: int = 0
    relevance_score: float = 0.0

class SearchRequest(BaseModel):
    """Search request model"""
    query: str
    filters: Optional[Dict] = {}
    page: int = 1
    page_size: int = 20
    sort_by: str = "relevance"

class SearchResponse(BaseModel):
    """Search response model"""
    query: str
    total_results: int
    page: int
    page_size: int
    results: List[Dict]
    facets: Dict
    search_time_ms: float
    suggestions: List[str] = []

class IntelligentSearchEngine:
    """Advanced search engine with multiple ranking algorithms"""
    
    def __init__(self, data_directory: str = "./data"):
        self.data_dir = Path(data_directory)
        self.papers: List[ResearchPaper] = []
        self.vocabulary: Set[str] = set()
        
        # Indexing structures
        self.term_document_matrix = None
        self.document_term_frequencies = {}
        self.inverse_document_frequencies = {}
        self.author_publication_map = defaultdict(list)
        self.year_publication_map = defaultdict(list)
        self.keyword_publication_map = defaultdict(set)
        
        # Enhanced TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=15000,
            ngram_range=(1, 3),
            stop_words='english',
            sublinear_tf=True,
            smooth_idf=True,
            use_idf=True,
            min_df=1,
            max_df=0.95
        )
        
        # Load and index data
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize search engine with data"""
        try:
            self._load_publications()
            self._build_indices()
            self._compute_term_statistics()
            logger.info(f"Engine initialized with {len(self.papers)} papers")
        except Exception as e:
            logger.error(f"Engine initialization failed: {e}")
            raise
    
    def _load_publications(self):
        """Load publication data from JSON files"""
        publications_file = self.data_dir / "publications.json"
        
        if not publications_file.exists():
            publications_file = self.data_dir / "publications_full.json"
        
        if publications_file.exists():
            with open(publications_file, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
                
            for item in raw_data:
                paper = ResearchPaper(
                    title=item.get('title', ''),
                    authors=item.get('authors', []),
                    link=item.get('link', ''),
                    published_date=item.get('published_date', ''),
                    abstract=item.get('abstract', ''),
                    author_profiles=item.get('author_profiles', {}),
                    keywords=self._extract_keywords_from_text(
                        item.get('title', '') + ' ' + item.get('abstract', '')
                    )
                )
                self.papers.append(paper)
            
            logger.info(f"Loaded {len(self.papers)} publications")
    
    def _extract_keywords_from_text(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        # Simple keyword extraction based on important terms
        text_lower = text.lower()
        
        # Remove common words and extract significant terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                     'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was',
                     'are', 'were', 'been', 'be', 'have', 'has', 'had', 'do',
                     'does', 'did', 'will', 'would', 'could', 'should', 'may',
                     'might', 'must', 'can', 'shall', 'this', 'that', 'these',
                     'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'}
        
        words = re.findall(r'\b[a-z]+\b', text_lower)
        keywords = [w for w in words if len(w) > 3 and w not in stop_words]
        
        # Get unique keywords
        return list(set(keywords))[:10]  # Limit to top 10 keywords
    
    def _build_indices(self):
        """Build search indices"""
        # Create document corpus for TF-IDF
        corpus = []
        
        for idx, paper in enumerate(self.papers):
            # Build document text with weighted components
            doc_text = ' '.join([
                paper.title * 3,  # Triple weight for title
                ' '.join(paper.authors) * 2,  # Double weight for authors
                paper.abstract,
                ' '.join(paper.keywords)
            ])
            corpus.append(doc_text)
            
            # Build author index
            for author in paper.authors:
                self.author_publication_map[author.lower()].append(idx)
            
            # Build year index
            year_match = re.search(r'\b(19|20)\d{2}\b', paper.published_date)
            if year_match:
                year = year_match.group()
                self.year_publication_map[year].append(idx)
            
            # Build keyword index
            for keyword in paper.keywords:
                self.keyword_publication_map[keyword].add(idx)
        
        # Create TF-IDF matrix
        self.term_document_matrix = self.vectorizer.fit_transform(corpus)
        
        logger.info(f"Built indices: {len(self.author_publication_map)} authors, "
                   f"{len(self.year_publication_map)} years, "
                   f"{len(self.keyword_publication_map)} keywords")
    
    def _compute_term_statistics(self):
        """Compute term frequency statistics"""
        # Extract vocabulary
        self.vocabulary = set(self.vectorizer.get_feature_names_out())
        
        # Calculate document frequencies
        for term in self.vocabulary:
            term_idx = self.vectorizer.vocabulary_.get(term)
            if term_idx is not None:
                df = np.sum(self.term_document_matrix[:, term_idx].toarray() > 0)
                self.inverse_document_frequencies[term] = math.log(
                    len(self.papers) / (df + 1)
                )
    
    def _calculate_bm25_score(self, query_terms: List[str], doc_idx: int, 
                             k1: float = 1.2, b: float = 0.75) -> float:
        """Calculate BM25 score for a document"""
        score = 0.0
        doc_vector = self.term_document_matrix[doc_idx].toarray().flatten()
        avg_doc_length = self.term_document_matrix.sum(axis=1).mean()
        doc_length = doc_vector.sum()
        
        for term in query_terms:
            if term in self.vectorizer.vocabulary_:
                term_idx = self.vectorizer.vocabulary_[term]
                tf = doc_vector[term_idx]
                
                if tf > 0:
                    idf = self.inverse_document_frequencies.get(term, 0)
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * (1 - b + b * doc_length / avg_doc_length)
                    score += idf * numerator / denominator
        
        return score
    
    def search(self, query: str, filters: Dict = None, 
              page: int = 1, page_size: int = 20) -> Dict:
        """Advanced search with multiple ranking algorithms"""
        start_time = datetime.now()
        
        # Process query
        query_lower = query.lower()
        query_terms = query_lower.split()
        
        # Transform query for TF-IDF
        query_vector = self.vectorizer.transform([query])
        
        # Calculate cosine similarity
        cosine_scores = cosine_similarity(query_vector, self.term_document_matrix).flatten()
        
        # Calculate BM25 scores
        bm25_scores = np.array([
            self._calculate_bm25_score(query_terms, idx)
            for idx in range(len(self.papers))
        ])
        
        # Combine scores (weighted average)
        combined_scores = 0.7 * cosine_scores + 0.3 * bm25_scores
        
        # Apply filters
        candidate_indices = self._apply_filters(filters)
        
        # Filter scores
        filtered_scores = [
            (idx, combined_scores[idx])
            for idx in candidate_indices
            if combined_scores[idx] > 0.01
        ]
        
        # Sort by score
        filtered_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Pagination
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        page_results = filtered_scores[start_idx:end_idx]
        
        # Build results
        results = []
        for idx, score in page_results:
            paper = self.papers[idx]
            results.append({
                'title': paper.title,
                'authors': paper.authors,
                'link': paper.link,
                'published_date': paper.published_date,
                'abstract': paper.abstract[:300] + '...' if len(paper.abstract) > 300 else paper.abstract,
                'author_profiles': paper.author_profiles,
                'keywords': paper.keywords[:5],
                'relevance_score': float(score)
            })
        
        # Generate facets
        facets = self._generate_facets(filtered_scores)
        
        # Generate suggestions
        suggestions = self._generate_suggestions(query, results[:5])
        
        # Calculate search time
        search_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return {
            'query': query,
            'total_results': len(filtered_scores),
            'page': page,
            'page_size': page_size,
            'results': results,
            'facets': facets,
            'suggestions': suggestions,
            'search_time_ms': search_time
        }
    
    def _apply_filters(self, filters: Optional[Dict]) -> Set[int]:
        """Apply search filters"""
        candidate_indices = set(range(len(self.papers)))
        
        if not filters:
            return candidate_indices
        
        # Author filter
        if 'author' in filters:
            author_indices = set()
            for author in self.author_publication_map:
                if filters['author'].lower() in author:
                    author_indices.update(self.author_publication_map[author])
            candidate_indices &= author_indices
        
        # Year filter
        if 'year' in filters:
            year_indices = set(self.year_publication_map.get(str(filters['year']), []))
            candidate_indices &= year_indices
        
        # Year range filter
        if 'year_from' in filters or 'year_to' in filters:
            year_range_indices = set()
            for year, indices in self.year_publication_map.items():
                year_int = int(year)
                if 'year_from' in filters and year_int < int(filters['year_from']):
                    continue
                if 'year_to' in filters and year_int > int(filters['year_to']):
                    continue
                year_range_indices.update(indices)
            candidate_indices &= year_range_indices
        
        return candidate_indices
    
    def _generate_facets(self, scored_results: List[Tuple[int, float]]) -> Dict:
        """Generate search facets for filtering"""
        facets = {
            'authors': defaultdict(int),
            'years': defaultdict(int),
            'keywords': defaultdict(int)
        }
        
        for idx, _ in scored_results[:100]:  # Analyze top 100 results
            paper = self.papers[idx]
            
            # Count authors
            for author in paper.authors:
                facets['authors'][author] += 1
            
            # Count years
            year_match = re.search(r'\b(19|20)\d{2}\b', paper.published_date)
            if year_match:
                facets['years'][year_match.group()] += 1
            
            # Count keywords
            for keyword in paper.keywords[:3]:
                facets['keywords'][keyword] += 1
        
        # Convert to sorted lists
        return {
            'authors': sorted(facets['authors'].items(), key=lambda x: x[1], reverse=True)[:10],
            'years': sorted(facets['years'].items(), key=lambda x: x[0], reverse=True),
            'keywords': sorted(facets['keywords'].items(), key=lambda x: x[1], reverse=True)[:15]
        }
    
    def _generate_suggestions(self, query: str, top_results: List[Dict]) -> List[str]:
        """Generate search suggestions based on results"""
        suggestions = set()
        
        # Extract frequent terms from top results
        for result in top_results:
            keywords = result.get('keywords', [])
            for keyword in keywords:
                if keyword not in query.lower():
                    suggestions.add(keyword)
        
        return list(suggestions)[:5]
    
    def get_statistics(self) -> Dict:
        """Get search engine statistics"""
        total_authors = set()
        years = []
        
        for paper in self.papers:
            total_authors.update(paper.authors)
            year_match = re.search(r'\b(19|20)\d{2}\b', paper.published_date)
            if year_match:
                years.append(int(year_match.group()))
        
        return {
            'total_publications': len(self.papers),
            'total_authors': len(total_authors),
            'total_keywords': len(self.keyword_publication_map),
            'year_range': f"{min(years) if years else 'N/A'} - {max(years) if years else 'N/A'}",
            'index_size': self.term_document_matrix.shape if self.term_document_matrix is not None else (0, 0),
            'last_updated': datetime.now().isoformat()
        }

# Initialize search engine
search_engine = IntelligentSearchEngine()

# API Endpoints
@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "service": "Academic Research Search Engine",
        "author": "Abishek Bimali",
        "version": "2.0.0",
        "endpoints": [
            "/search - Advanced search endpoint",
            "/statistics - Engine statistics",
            "/publications - Browse publications",
            "/authors - List all authors",
            "/trending - Trending research topics"
        ]
    }

@app.post("/search", response_model=SearchResponse)
async def search_publications(request: SearchRequest):
    """Advanced search endpoint"""
    try:
        results = search_engine.search(
            query=request.query,
            filters=request.filters,
            page=request.page,
            page_size=request.page_size
        )
        return SearchResponse(**results)
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search")
async def search_get(
    q: str = Query(..., description="Search query"),
    author: Optional[str] = Query(None, description="Filter by author"),
    year: Optional[int] = Query(None, description="Filter by year"),
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100)
):
    """GET search endpoint for compatibility"""
    filters = {}
    if author:
        filters['author'] = author
    if year:
        filters['year'] = year
    
    results = search_engine.search(q, filters, page, limit)
    return JSONResponse(content=results)

@app.get("/statistics")
async def get_statistics():
    """Get engine statistics"""
    return search_engine.get_statistics()

@app.get("/publications")
async def browse_publications(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100)
):
    """Browse all publications with pagination"""
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    
    publications = []
    for paper in search_engine.papers[start_idx:end_idx]:
        publications.append({
            'title': paper.title,
            'authors': paper.authors,
            'link': paper.link,
            'published_date': paper.published_date,
            'abstract': paper.abstract[:200] + '...' if len(paper.abstract) > 200 else paper.abstract
        })
    
    return {
        'total': len(search_engine.papers),
        'page': page,
        'limit': limit,
        'publications': publications
    }

@app.get("/authors")
async def list_authors():
    """List all unique authors"""
    authors = set()
    for paper in search_engine.papers:
        authors.update(paper.authors)
    
    sorted_authors = sorted(list(authors))
    
    return {
        'total': len(sorted_authors),
        'authors': sorted_authors
    }

@app.get("/trending")
async def trending_topics():
    """Get trending research topics"""
    current_year = datetime.now().year
    recent_years = {str(current_year - i) for i in range(3)}
    
    keyword_counts = defaultdict(int)
    
    for paper in search_engine.papers:
        year_match = re.search(r'\b(19|20)\d{2}\b', paper.published_date)
        if year_match and year_match.group() in recent_years:
            for keyword in paper.keywords:
                keyword_counts[keyword] += 1
    
    # Sort by count and get top 20
    trending = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    
    return {
        'period': f"{min(recent_years)}-{max(recent_years)}",
        'trending_topics': [
            {'keyword': kw, 'frequency': count}
            for kw, count in trending
        ]
    }

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )