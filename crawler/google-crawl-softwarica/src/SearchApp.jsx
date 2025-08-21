import React, { useState, useEffect } from 'react';
import './SearchApp.css';

const SearchApp = () => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState(null);
  const [searchTime, setSearchTime] = useState(null);
  const [authorFilter, setAuthorFilter] = useState('');
  const [yearFilter, setYearFilter] = useState('');
  const [showFilters, setShowFilters] = useState(false);
  const [allAuthors, setAllAuthors] = useState([]);

  const API_BASE_URL = 'http://localhost:8000';

  useEffect(() => {
    fetchStats();
    fetchAuthors();
  }, []);

  const fetchStats = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/stats`);
      const data = await response.json();
      setStats(data);
    } catch (error) {
      console.error('Error fetching stats:', error);
    }
  };

  const fetchAuthors = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/authors`);
      const data = await response.json();
      setAllAuthors(data.authors);
    } catch (error) {
      console.error('Error fetching authors:', error);
    }
  };

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setResults([]);

    try {
      const params = new URLSearchParams({
        q: query,
        limit: 50
      });

      if (authorFilter) params.append('author', authorFilter);
      if (yearFilter) params.append('year', yearFilter);

      const response = await fetch(`${API_BASE_URL}/search?${params}`);
      const data = await response.json();
      
      setResults(data.results);
      setSearchTime(data.search_time);
    } catch (error) {
      console.error('Search error:', error);
      alert('Error performing search. Please check if the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const highlightQuery = (text) => {
    if (!query) return text;
    const parts = text.split(new RegExp(`(${query})`, 'gi'));
    return parts.map((part, i) => 
      part.toLowerCase() === query.toLowerCase() ? 
        <mark key={i}>{part}</mark> : part
    );
  };

  const formatDate = (dateStr) => {
    return dateStr || 'Date not available';
  };

  const truncateAbstract = (abstract, maxLength = 300) => {
    if (!abstract) return 'No abstract available';
    if (abstract.length <= maxLength) return abstract;
    return abstract.substr(0, maxLength) + '...';
  };

  return (
    <div className="search-app">
      <header className="search-header">
        <h1>Coventry University Publications Search</h1>
        <p className="subtitle">School of Economics, Finance and Accounting</p>
        {stats && (
          <div className="stats-bar">
            <span>📄 {stats.total_publications} Publications</span>
            <span>👥 {stats.total_unique_authors} Authors</span>
            <span>📅 {stats.years_range}</span>
          </div>
        )}
      </header>

      <main className="search-main">
        <form onSubmit={handleSearch} className="search-form">
          <div className="search-box">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search for publications, authors, keywords..."
              className="search-input"
            />
            <button type="submit" className="search-button" disabled={loading}>
              {loading ? 'Searching...' : '🔍 Search'}
            </button>
          </div>

          <button 
            type="button" 
            className="filter-toggle"
            onClick={() => setShowFilters(!showFilters)}
          >
            {showFilters ? '▼' : '▶'} Advanced Filters
          </button>

          {showFilters && (
            <div className="filters">
              <div className="filter-group">
                <label>Author:</label>
                <input
                  type="text"
                  value={authorFilter}
                  onChange={(e) => setAuthorFilter(e.target.value)}
                  placeholder="Filter by author name"
                  list="authors-list"
                />
                <datalist id="authors-list">
                  {allAuthors.slice(0, 100).map((author, idx) => (
                    <option key={idx} value={author} />
                  ))}
                </datalist>
              </div>
              <div className="filter-group">
                <label>Year:</label>
                <input
                  type="text"
                  value={yearFilter}
                  onChange={(e) => setYearFilter(e.target.value)}
                  placeholder="e.g., 2024"
                  maxLength="4"
                />
              </div>
            </div>
          )}
        </form>

        {searchTime !== null && results.length > 0 && (
          <div className="search-info">
            Found {results.length} results in {searchTime.toFixed(3)} seconds
          </div>
        )}

        {loading && (
          <div className="loading">
            <div className="spinner"></div>
            <p>Searching publications...</p>
          </div>
        )}

        {!loading && results.length > 0 && (
          <div className="results">
            {results.map((pub, index) => (
              <article key={index} className="result-item">
                <h3 className="result-title">
                  <a href={pub.link} target="_blank" rel="noopener noreferrer">
                    {highlightQuery(pub.title)}
                  </a>
                </h3>
                
                <div className="result-meta">
                  <span className="authors">
                    {pub.authors.map((author, idx) => {
                      const profileUrl = pub.author_profiles && pub.author_profiles[author];
                      console.log(`Author: ${author}, Profile URL:`, profileUrl); 
                      return (
                        <span key={idx}>
                          {profileUrl ? (
                            <a 
                              href={profileUrl}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="author-link"
                              title={`View ${author}'s profile`}
                            >
                              {author}
                            </a>
                          ) : (
                            <span>{author}</span>
                          )}
                          {idx < pub.authors.length - 1 && ', '}
                        </span>
                      );
                    })}
                  </span>
                  <span className="date">📅 {formatDate(pub.published_date)}</span>
                  {pub.score && (
                    <span className="relevance">
                      Relevance: {(pub.score * 100).toFixed(1)}%
                    </span>
                  )}
                </div>

                <p className="result-abstract">
                  {highlightQuery(truncateAbstract(pub.abstract))}
                </p>

                <div className="result-actions">
                  <a 
                    href={pub.link} 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="view-publication"
                  >
                    View Publication →
                  </a>
                </div>
              </article>
            ))}
          </div>
        )}

        {!loading && query && results.length === 0 && (
          <div className="no-results">
            <p>No publications found for "{query}"</p>
            <p>Try different keywords or check the filters</p>
          </div>
        )}
      </main>

      <footer className="search-footer">
        <p>Prepared by Abishek Bimali | Data from Coventry University Pure Portal</p>
      </footer>
    </div>
  );
};

export default SearchApp;