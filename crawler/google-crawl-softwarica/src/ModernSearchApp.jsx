/**
 * Modern Academic Search Interface
 * Author: Abishek Bimali
 * Date: 2025
 */

import React, { useState, useEffect, useCallback } from 'react';
import './ModernSearchApp.css';

const ModernSearchApp = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);
  const [statistics, setStatistics] = useState(null);
  const [activeFilters, setActiveFilters] = useState({});
  const [searchFacets, setSearchFacets] = useState(null);
  const [viewMode, setViewMode] = useState('grid'); // grid or list
  const [searchSuggestions, setSuggestions] = useState([]);
  const [currentPage, setCurrentPage] = useState(1);
  const [totalResults, setTotalResults] = useState(0);
  const [searchTime, setSearchTime] = useState(0);

  const API_BASE = 'http://localhost:8000';

  useEffect(() => {
    loadStatistics();
  }, []);

  const loadStatistics = async () => {
    try {
      const response = await fetch(`${API_BASE}/statistics`);
      const data = await response.json();
      setStatistics(data);
    } catch (error) {
      console.error('Failed to load statistics:', error);
    }
  };

  const performSearch = useCallback(async (query, page = 1) => {
    if (!query.trim()) return;
    
    setIsSearching(true);
    setCurrentPage(page);
    
    try {
      const params = new URLSearchParams({
        q: query,
        page: page,
        limit: 12,
        ...activeFilters
      });
      
      const response = await fetch(`${API_BASE}/search?${params}`);
      const data = await response.json();
      
      setSearchResults(data.results || []);
      setTotalResults(data.total_results || 0);
      setSearchFacets(data.facets || {});
      setSuggestions(data.suggestions || []);
      setSearchTime(data.search_time_ms || 0);
    } catch (error) {
      console.error('Search failed:', error);
      setSearchResults([]);
    } finally {
      setIsSearching(false);
    }
  }, [activeFilters]);

  const handleSearch = (e) => {
    e.preventDefault();
    performSearch(searchQuery, 1);
  };

  const handleFilterChange = (filterType, value) => {
    const newFilters = { ...activeFilters };
    if (value) {
      newFilters[filterType] = value;
    } else {
      delete newFilters[filterType];
    }
    setActiveFilters(newFilters);
    if (searchQuery) {
      performSearch(searchQuery, 1);
    }
  };

  const highlightQuery = (text, query) => {
    if (!query || !text) return text;
    
    const terms = query.toLowerCase().split(' ').filter(t => t.length > 2);
    let highlightedText = text;
    
    terms.forEach(term => {
      const regex = new RegExp(`(${term})`, 'gi');
      highlightedText = highlightedText.replace(regex, '<span class="highlight">$1</span>');
    });
    
    return <span dangerouslySetInnerHTML={{ __html: highlightedText }} />;
  };

  const formatDate = (dateString) => {
    if (!dateString) return 'Unknown date';
    const match = dateString.match(/\b(19|20)\d{2}\b/);
    return match ? match[0] : dateString;
  };

  const ResultCard = ({ paper }) => (
    <article className={`result-card ${viewMode}`}>
      <div className="card-header">
        <h3 className="paper-title">
          <a href={paper.link} target="_blank" rel="noopener noreferrer">
            {highlightQuery(paper.title, searchQuery)}
          </a>
        </h3>
        <span className="relevance-badge">
          {Math.round(paper.relevance_score * 100)}%
        </span>
      </div>
      
      <div className="paper-authors">
        {paper.authors.map((author, idx) => (
          <span key={idx} className="author-tag">
            {author}
            {paper.author_profiles && paper.author_profiles[author] && (
              <a 
                href={paper.author_profiles[author]} 
                target="_blank" 
                rel="noopener noreferrer"
                className="author-link"
              >
                →
              </a>
            )}
          </span>
        ))}
      </div>
      
      <div className="paper-metadata">
        <span className="pub-date">📅 {formatDate(paper.published_date)}</span>
      </div>
      
      <p className="paper-abstract">
        {highlightQuery(paper.abstract, searchQuery)}
      </p>
      
      {paper.keywords && paper.keywords.length > 0 && (
        <div className="paper-keywords">
          {paper.keywords.map((keyword, idx) => (
            <span key={idx} className="keyword-chip">
              {keyword}
            </span>
          ))}
        </div>
      )}
    </article>
  );

  return (
    <div className="modern-search-app">
      <header className="app-header">
        <div className="header-content">
          <h1 className="app-title">
            📚 Academic Research Explorer
          </h1>
          <p className="app-subtitle">
            Intelligent Search for Academic Publications
          </p>
          {statistics && (
            <div className="stats-bar">
              <span className="stat-item">
                <strong>{statistics.total_publications}</strong> Publications
              </span>
              <span className="stat-item">
                <strong>{statistics.total_authors}</strong> Authors
              </span>
              <span className="stat-item">
                <strong>{statistics.total_keywords}</strong> Keywords
              </span>
              <span className="stat-item">
                Years: <strong>{statistics.year_range}</strong>
              </span>
            </div>
          )}
        </div>
      </header>

      <section className="search-section">
        <form onSubmit={handleSearch} className="search-form">
          <div className="search-input-wrapper">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search papers, authors, topics..."
              className="search-input"
              autoFocus
            />
            <button 
              type="submit" 
              className="search-button"
              disabled={isSearching}
            >
              {isSearching ? (
                <span className="loading-spinner">⟳</span>
              ) : (
                <span>Search</span>
              )}
            </button>
          </div>
          
          {searchSuggestions.length > 0 && (
            <div className="suggestions">
              <span className="suggestion-label">Try also:</span>
              {searchSuggestions.map((suggestion, idx) => (
                <button
                  key={idx}
                  type="button"
                  className="suggestion-chip"
                  onClick={() => {
                    setSearchQuery(searchQuery + ' ' + suggestion);
                    performSearch(searchQuery + ' ' + suggestion, 1);
                  }}
                >
                  + {suggestion}
                </button>
              ))}
            </div>
          )}
        </form>
      </section>

      <main className="results-section">
        {searchResults.length > 0 && (
          <>
            <div className="results-header">
              <div className="results-info">
                <h2 className="results-count">
                  {totalResults} results
                  <span className="search-time">
                    ({searchTime.toFixed(0)}ms)
                  </span>
                </h2>
              </div>
              
              <div className="view-controls">
                <button
                  className={`view-btn ${viewMode === 'grid' ? 'active' : ''}`}
                  onClick={() => setViewMode('grid')}
                >
                  ⊞ Grid
                </button>
                <button
                  className={`view-btn ${viewMode === 'list' ? 'active' : ''}`}
                  onClick={() => setViewMode('list')}
                >
                  ☰ List
                </button>
              </div>
            </div>

            <div className="main-content">
              {searchFacets && Object.keys(searchFacets).length > 0 && (
                <aside className="facets-sidebar">
                  <h3>Refine Results</h3>
                  
                  {searchFacets.authors && searchFacets.authors.length > 0 && (
                    <div className="facet-group">
                      <h4>Top Authors</h4>
                      <div className="facet-items">
                        {searchFacets.authors.slice(0, 5).map(([author, count]) => (
                          <button
                            key={author}
                            className="facet-item"
                            onClick={() => handleFilterChange('author', author)}
                          >
                            {author} ({count})
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  {searchFacets.years && searchFacets.years.length > 0 && (
                    <div className="facet-group">
                      <h4>Publication Years</h4>
                      <div className="facet-items">
                        {searchFacets.years.slice(0, 5).map(([year, count]) => (
                          <button
                            key={year}
                            className="facet-item"
                            onClick={() => handleFilterChange('year', year)}
                          >
                            {year} ({count})
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  {searchFacets.keywords && searchFacets.keywords.length > 0 && (
                    <div className="facet-group">
                      <h4>Related Topics</h4>
                      <div className="facet-items">
                        {searchFacets.keywords.slice(0, 8).map(([keyword, count]) => (
                          <button
                            key={keyword}
                            className="facet-item keyword"
                            onClick={() => {
                              setSearchQuery(searchQuery + ' ' + keyword);
                              performSearch(searchQuery + ' ' + keyword, 1);
                            }}
                          >
                            {keyword} ({count})
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  {Object.keys(activeFilters).length > 0 && (
                    <button 
                      className="clear-filters-btn"
                      onClick={() => {
                        setActiveFilters({});
                        performSearch(searchQuery, 1);
                      }}
                    >
                      Clear All Filters
                    </button>
                  )}
                </aside>
              )}
              
              <div className={`results-container ${viewMode}`}>
                {searchResults.map((paper, idx) => (
                  <ResultCard key={idx} paper={paper} />
                ))}
              </div>
            </div>

            {totalResults > 12 && (
              <div className="pagination">
                <button
                  disabled={currentPage === 1}
                  onClick={() => performSearch(searchQuery, currentPage - 1)}
                  className="pagination-btn"
                >
                  Previous
                </button>
                <span className="page-info">
                  Page {currentPage} of {Math.ceil(totalResults / 12)}
                </span>
                <button
                  disabled={currentPage >= Math.ceil(totalResults / 12)}
                  onClick={() => performSearch(searchQuery, currentPage + 1)}
                  className="pagination-btn"
                >
                  Next
                </button>
              </div>
            )}
          </>
        )}
        
        {searchQuery && searchResults.length === 0 && !isSearching && (
          <div className="no-results">
            <h2>No results found</h2>
            <p>Try adjusting your search terms or removing filters</p>
          </div>
        )}
        
        {!searchQuery && !isSearching && (
          <div className="welcome-message">
            <h2>Welcome to Academic Research Explorer</h2>
            <p>Start searching to discover academic publications</p>
            <div className="quick-actions">
              <h3>Quick Search Ideas:</h3>
              <div className="quick-search-chips">
                <button
                  onClick={() => {
                    setSearchQuery('machine learning');
                    performSearch('machine learning', 1);
                  }}
                  className="quick-chip"
                >
                  Machine Learning
                </button>
                <button
                  onClick={() => {
                    setSearchQuery('climate change');
                    performSearch('climate change', 1);
                  }}
                  className="quick-chip"
                >
                  Climate Change
                </button>
                <button
                  onClick={() => {
                    setSearchQuery('artificial intelligence');
                    performSearch('artificial intelligence', 1);
                  }}
                  className="quick-chip"
                >
                  Artificial Intelligence
                </button>
                <button
                  onClick={() => {
                    setSearchQuery('covid-19');
                    performSearch('covid-19', 1);
                  }}
                  className="quick-chip"
                >
                  COVID-19
                </button>
              </div>
            </div>
          </div>
        )}
      </main>

      <footer className="app-footer">
        <p>
          Academic Research Explorer v2.0 | Created by <strong>Abishek Bimali</strong> | {new Date().getFullYear()}
        </p>
      </footer>
    </div>
  );
};

export default ModernSearchApp;