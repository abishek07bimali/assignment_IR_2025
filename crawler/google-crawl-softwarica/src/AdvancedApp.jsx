import { useState, useEffect } from 'react'
import './App.css'

function AdvancedApp() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [stats, setStats] = useState(null)
  const [authors, setAuthors] = useState([])
  
  // Filters
  const [selectedAuthor, setSelectedAuthor] = useState('')
  const [yearFrom, setYearFrom] = useState('')
  const [yearTo, setYearTo] = useState('')
  const [deptOnly, setDeptOnly] = useState(false)
  const [showFilters, setShowFilters] = useState(false)

  useEffect(() => {
    fetchStats()
    fetchAuthors()
  }, [])

  const fetchStats = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/stats')
      const data = await response.json()
      setStats(data)
    } catch (error) {
      console.error('Error fetching stats:', error)
    }
  }

  const fetchAuthors = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/authors')
      const data = await response.json()
      setAuthors(data.authors)
    } catch (error) {
      console.error('Error fetching authors:', error)
    }
  }

  const handleSearch = async (e) => {
    e.preventDefault()
    setLoading(true)
    
    try {
      const params = new URLSearchParams({
        q: query,
        limit: 20,
        ...(selectedAuthor && { author: selectedAuthor }),
        ...(yearFrom && { year_from: yearFrom }),
        ...(yearTo && { year_to: yearTo }),
        dept_only: deptOnly
      })
      
      const response = await fetch(`http://localhost:8000/api/search?${params}`)
      const data = await response.json()
      setResults(data.results)
    } catch (error) {
      console.error('Search error:', error)
      setResults([])
    } finally {
      setLoading(false)
    }
  }

  const clearFilters = () => {
    setSelectedAuthor('')
    setYearFrom('')
    setYearTo('')
    setDeptOnly(false)
  }

  const highlightText = (text, searchQuery) => {
    if (!searchQuery || !text) return text
    
    const words = searchQuery.toLowerCase().split(' ').filter(w => w.length > 2)
    let highlightedText = text
    
    words.forEach(word => {
      const regex = new RegExp(`(${word})`, 'gi')
      highlightedText = highlightedText.replace(regex, '<mark>$1</mark>')
    })
    
    return <span dangerouslySetInnerHTML={{ __html: highlightedText }} />
  }

  const formatAuthors = (authors) => {
    if (!authors || !Array.isArray(authors)) return ''
    
    if (authors.length > 0 && typeof authors[0] === 'object') {
      return authors.map((author, idx) => (
        <span key={idx}>
          {author.profile_url ? (
            <>
              <a href={author.profile_url} target="_blank" rel="noopener noreferrer" 
                 style={{color: author.is_dept_member ? '#006621' : '#5f6368'}}>
                {author.name}
              </a>
              {author.is_dept_member && <span title="Department Member"> ⭐</span>}
            </>
          ) : (
            <span style={{color: author.is_dept_member ? '#006621' : '#5f6368'}}>
              {author.name}
              {author.is_dept_member && <span title="Department Member"> ⭐</span>}
            </span>
          )}
          {idx < authors.length - 1 && ', '}
        </span>
      ))
    } else {
      return authors.join(', ')
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1>Coventry University Publications Search</h1>
        <p className="subtitle">School of Economics, Finance and Accounting</p>
        {stats && (
          <p className="stats">
            {stats.total_publications} publications | {stats.total_authors} authors | {stats.dept_members} department members
          </p>
        )}
      </header>

      <form onSubmit={handleSearch} className="search-form">
        <div className="search-container">
          <div className="search-box">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search publications, authors, keywords..."
              className="search-input"
            />
            <button type="submit" className="search-button" disabled={loading}>
              {loading ? 'Searching...' : 'Search'}
            </button>
          </div>
          
          <button 
            type="button" 
            className="filter-toggle"
            onClick={() => setShowFilters(!showFilters)}
          >
            {showFilters ? '▼' : '▶'} Advanced Filters
          </button>
        </div>

        {showFilters && (
          <div className="filters">
            <div className="filter-group">
              <label>Author:</label>
              <select 
                value={selectedAuthor} 
                onChange={(e) => setSelectedAuthor(e.target.value)}
                className="filter-select"
              >
                <option value="">All Authors</option>
                {authors.filter(a => a.is_dept_member).map(author => (
                  <option key={author.name} value={author.name}>
                    {author.name} ({author.count}) ⭐
                  </option>
                ))}
                <option disabled>──────────</option>
                {authors.filter(a => !a.is_dept_member).slice(0, 20).map(author => (
                  <option key={author.name} value={author.name}>
                    {author.name} ({author.count})
                  </option>
                ))}
              </select>
            </div>

            <div className="filter-group">
              <label>Year From:</label>
              <input 
                type="number" 
                value={yearFrom}
                onChange={(e) => setYearFrom(e.target.value)}
                placeholder="2000"
                min="1990"
                max={new Date().getFullYear()}
                className="filter-input"
              />
            </div>

            <div className="filter-group">
              <label>Year To:</label>
              <input 
                type="number" 
                value={yearTo}
                onChange={(e) => setYearTo(e.target.value)}
                placeholder={new Date().getFullYear()}
                min="1990"
                max={new Date().getFullYear()}
                className="filter-input"
              />
            </div>

            <div className="filter-group">
              <label>
                <input 
                  type="checkbox"
                  checked={deptOnly}
                  onChange={(e) => setDeptOnly(e.target.checked)}
                />
                Department Members Only
              </label>
            </div>

            <button type="button" onClick={clearFilters} className="clear-filters">
              Clear Filters
            </button>
          </div>
        )}
      </form>

      <div className="results-container">
        {results.length > 0 && (
          <p className="results-count">
            Found {results.length} results
            {(selectedAuthor || yearFrom || yearTo || deptOnly) && ' (filtered)'}
          </p>
        )}
        
        {results.map((pub, index) => (
          <article key={pub.id || index} className="result-item">
            <h3 className="result-title">
              <a href={pub.link} target="_blank" rel="noopener noreferrer">
                {highlightText(pub.title, query)}
              </a>
            </h3>
            
            <div className="result-authors">
              {formatAuthors(pub.authors)}
            </div>
            
            <div className="result-metadata">
              {pub.published_date && (
                <span className="result-date">Published: {pub.published_date}</span>
              )}
              {pub.year && (
                <span className="result-year"> ({pub.year})</span>
              )}
              {pub.type && (
                <span className="result-type"> • {pub.type}</span>
              )}
              {pub.journal && (
                <span className="result-journal"> • {pub.journal}</span>
              )}
            </div>
            
            {pub.abstract && (
              <div className="result-abstract">
                {highlightText(pub.abstract.substring(0, 300), query)}
                {pub.abstract.length > 300 && '...'}
              </div>
            )}
            
            {pub.keywords && pub.keywords.length > 0 && (
              <div className="result-keywords">
                Keywords: {pub.keywords.map((kw, idx) => (
                  <span key={idx} className="keyword">
                    {kw}{idx < pub.keywords.length - 1 && ', '}
                  </span>
                ))}
              </div>
            )}
            
            {pub.relevance_score && (
              <div className="relevance-score">
                Relevance: {(pub.relevance_score * 100).toFixed(1)}%
              </div>
            )}
          </article>
        ))}
        
        {query && results.length === 0 && !loading && (
          <p className="no-results">No results found for "{query}"</p>
        )}
      </div>
    </div>
  )
}

export default AdvancedApp