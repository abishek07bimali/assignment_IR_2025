# 🔍 Academic Research Search Platform

**Developer:** Abishek Bimali  
**Project Type:** Information Retrieval System  

## Overview
A web-based academic publication search system with ranking algorithm. Provides fast, relevant search results with faceted filtering.

## Structure
```
crawler/
├── academic_search_engine.py    # Backend API server
├── data/
│   └── publications.json        # Publication database
├── google-crawl-softwarica/     # React frontend
│   ├── src/
│   │   ├── ModernSearchApp.jsx
│   │   └── main.jsx
│   └── package.json
└── requirements.txt
```

## Running Instructions

### Backend
```bash
# Install dependencies
pip install -r requirements.txt

# Start server
python academic_search_engine.py
# Runs on http://localhost:8000
```

### Frontend
```bash
# Navigate to frontend
cd google-crawl-softwarica/

# Install dependencies
npm install

# Start development server
npm run dev
# Runs on http://localhost:5173
```

## API Endpoints
- `GET /search?q=<query>` - Search publications
- `GET /statistics` - Get database statistics
- `GET /publications` - Browse all publications