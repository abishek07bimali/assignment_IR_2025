# Publications Crawler & Search Engine

**Author:** Abishek Bimali

## Project Overview

A comprehensive web crawler and search engine system designed to harvest academic publications from Coventry University's Pure Portal and provide an intelligent search interface for exploring the collected research data.

## System Architecture

The project consists of three main components:

1. **Web Crawler** - Python-based scraper for harvesting publication data
2. **Search Backend** - FastAPI server with TF-IDF search algorithm
3. **Search Frontend** - React with vite-based user interface

## Working Mechanism

### 1. Data Collection Phase

#### Web Crawler (`crawler.ipynb` / `assignment.py`)
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
   python -m uvicorn search_engine_backend:app --host 0.0.0.0 --port 8000

   # Server runs on http://localhost:8000
   ```

2. **Start Frontend Application**
   ```bash
   cd google-crawl-softwarica
   npm run dev
   # Frontend runs on http://localhost:5173
   ```

3. **Quick Start Script for linux**
   ```bash
   ./run_search_engine.sh
   ```




### TF-IDF (Term Frequency-Inverse Document Frequency)
- Measures importance of a term in a document relative to a collection
- Formula: `TF-IDF = TF(t,d) × IDF(t)`
- Used for converting text documents to numerical vectors

### Cosine Similarity
- Measures similarity between query vector and document vectors
- Range: 0 (no similarity) to 1 (identical)
- Formula: `similarity = cos(θ) = (A·B)/(||A||×||B||)`

---

**Developed by:** Abishek Bimali  
**Institution:** Coventry University