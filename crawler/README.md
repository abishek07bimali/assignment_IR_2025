# Publications Crawler & Search Engine

**Author:** Abishek Bimali

## Project Overview

A comprehensive web crawler and search engine system designed to harvest academic publications from Coventry University's Pure Portal and provide an intelligent search interface for exploring the collected research data.

## System Architecture

The project consists of three main components:

1. **Web Crawler** - Python-based scraper for harvesting publication data
2. **Search Backend** - FastAPI server with TF-IDF search algorithm
3. **Search Frontend** - React-based user interface

## Working Mechanism

### 1. Data Collection Phase

#### Web Crawler (`crawler.ipynb` / `ass2.py`)
The crawler systematically extracts publication data from the university portal:

- **Target URL**: Coventry University Pure Portal (School of Economics, Finance and Accounting)
- **Data Extracted**:
  - Publication titles
  - Author names and profiles
  - Publication dates
  - Abstracts
  - Publication links
  - Author profile links

- **Output Files**:
  - `data/publications.json` - Main publication dataset (770+ publications)
  - `data/publications_links.json` - Publication URLs for reference

### 2. Search Engine Backend

#### Core Components (`search_engine_backend.py`)

**Data Processing Pipeline:**

1. **Data Loading**
   - Reads publications from `data/publications.json`
   - Loads author profiles and department member information
   - Initializes in-memory data structures

2. **Indexing System**
   - **TF-IDF Matrix**: Creates vector representations of all documents
     - Max features: 10,000 terms
     - N-gram range: 1-3 (unigrams, bigrams, trigrams)
     - Sublinear term frequency for better weighting
   - **Inverted Indices**: 
     - Author Index: Maps authors to publication IDs
     - Year Index: Maps publication years to document IDs  
     - Keyword Index: Maps keywords to publication IDs (9,713+ keywords indexed)

3. **Search Algorithm**
   ```
   Query Processing:
   1. Preprocess query text (lowercase, remove punctuation)
   2. Duplicate important terms for emphasis
   3. Transform query to TF-IDF vector
   4. Calculate cosine similarity with all documents
   5. Apply keyword boosting for exact matches
   6. Filter by author/year if specified
   7. Rank and return top results
   ```

**API Endpoints:**
- `GET /` - API information
- `GET /search` - Main search endpoint with filters
- `GET /search/advanced` - Field-specific search
- `GET /stats` - Database statistics
- `GET /publications` - Paginated publication list
- `GET /authors` - List all authors
- `GET /trending` - Trending research topics
- `GET /reindex` - Rebuild search indices

### 3. Search Frontend

#### React Interface (`google-crawl-softwarica/src/SearchApp.jsx`)

**Features:**
- Real-time search with highlighting
- Advanced filters (author, year)
- Author autocomplete suggestions
- Relevance scoring display
- Responsive design
- Direct links to publications and author profiles

**User Interaction Flow:**
1. User enters search query
2. Frontend sends request to backend API
3. Backend processes query through TF-IDF algorithm
4. Results ranked by relevance score
5. Frontend displays results with metadata
6. Users can click through to original publications

## Technical Implementation

### Search Relevance Scoring

The system uses a hybrid scoring approach:

1. **TF-IDF Score** (Primary)
   - Measures term frequency and inverse document frequency
   - Gives higher weight to rare, distinctive terms
   - Title and author fields weighted 2x for importance

2. **Keyword Boost** (Secondary)
   - Exact keyword matches receive 0.1 boost per match
   - Ensures relevant documents aren't missed

3. **Combined Score**
   ```python
   final_score = tfidf_score + keyword_boost
   ```

### Performance Optimizations

- **Inverted Indices**: O(1) lookup for author/year filters
- **Keyword Index**: Fast keyword-based retrieval
- **Vector Caching**: Pre-computed TF-IDF vectors
- **Lazy Loading**: Pagination for large result sets

## Installation & Usage

### Prerequisites
```bash
# Python packages
pip install fastapi uvicorn scikit-learn numpy pydantic

# Node packages (for frontend)
cd google-crawl-softwarica
npm install
```

### Running the System

1. **Start Backend Server**
   ```bash
   python3 search_engine_backend.py
   # Server runs on http://localhost:8000
   ```

2. **Start Frontend Application**
   ```bash
   cd google-crawl-softwarica
   npm run dev
   # Frontend runs on http://localhost:5173
   ```

3. **Quick Start Script**
   ```bash
   ./run_search_engine.sh
   ```

## Data Statistics

- **Total Publications**: 770
- **Unique Authors**: 795
- **Year Range**: 1992-2025
- **Indexed Keywords**: 9,713
- **Average Search Time**: ~2ms

## API Usage Examples

### Basic Search
```bash
curl "http://localhost:8000/search?q=machine%20learning&limit=10"
```

### Filtered Search
```bash
curl "http://localhost:8000/search?q=finance&author=Hassan&year=2024"
```

### Advanced Search
```bash
curl "http://localhost:8000/search/advanced?title=blockchain&year_from=2020"
```

## Project Structure
```
crawler/
├── data/
│   ├── publications.json       # Main publication dataset
│   └── publications_links.json # Publication URLs
├── google-crawl-softwarica/    # React frontend
│   ├── src/
│   │   ├── SearchApp.jsx      # Main search component
│   │   └── SearchApp.css      # Styling
│   └── package.json
├── search_engine_backend.py    # FastAPI search server
├── crawler.ipynb               # Jupyter notebook crawler
├── ass2.py                     # Python crawler script
└── README.md                   # This file
```

## Key Algorithms

### TF-IDF (Term Frequency-Inverse Document Frequency)
- Measures importance of a term in a document relative to a collection
- Formula: `TF-IDF = TF(t,d) × IDF(t)`
- Used for converting text documents to numerical vectors

### Cosine Similarity
- Measures similarity between query vector and document vectors
- Range: 0 (no similarity) to 1 (identical)
- Formula: `similarity = cos(θ) = (A·B)/(||A||×||B||)`

## Future Enhancements

1. **Machine Learning Integration**
   - Neural network-based semantic search
   - Query expansion using word embeddings
   - Learning-to-rank algorithms

2. **Advanced Features**
   - Citation network analysis
   - Co-author collaboration graphs
   - Research trend predictions
   - Export functionality (CSV, BibTeX)

3. **Performance Improvements**
   - Elasticsearch integration
   - Redis caching layer
   - Distributed crawling

## License

This project is developed for academic purposes at Coventry University.

---

**Developed by:** Abishek Bimali  
**Institution:** Coventry University  
**Department:** School of Economics, Finance and Accounting