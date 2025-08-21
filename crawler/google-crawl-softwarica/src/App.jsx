import { useState, useEffect } from 'react'
import './App.css'

function App() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [stats, setStats] = useState(null)

  useEffect(() => {
    fetchStats()
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

  const handleSearch = async (e) => {
    e.preventDefault()
    setLoading(true)
    
    try {
      const response = await fetch(`http://localhost:8000/api/search?q=${encodeURIComponent(query)}&limit=20`)
      const data = await response.json()
      setResults(data.results)
    } catch (error) {
      console.error('Search error:', error)
      setResults([])
    } finally {
      setLoading(false)
    }
  }

  const highlightText = (text, searchQuery) => {
    if (!searchQuery) return text
    
    const words = searchQuery.toLowerCase().split(' ').filter(w => w.length > 2)
    let highlightedText = text
    
    words.forEach(word => {
      const regex = new RegExp(`(${word})`, 'gi')
      highlightedText = highlightedText.replace(regex, '<mark>$1</mark>')
    })
    
    return <span dangerouslySetInnerHTML={{ __html: highlightedText }} />
  }

  return (
    <div className="app">
      <header className="header">
        <h1>Department Publications Search</h1>
        {stats && (
          <p className="stats">
            {stats.total_publications} publications from {stats.total_authors} authors
          </p>
        )}
      </header>

      <form onSubmit={handleSearch} className="search-form">
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
      </form>

      <div className="results-container">
        {results.length > 0 && (
          <p className="results-count">About {results.length} results</p>
        )}
        
        {results.map((pub, index) => (
          <article key={index} className="result-item">
            <h3 className="result-title">
              <a href={pub.link} target="_blank" rel="noopener noreferrer">
                {highlightText(pub.title, query)}
              </a>
            </h3>
            
            <div className="result-authors">
              {pub.authors.join(', ')}
            </div>
            
            <div className="result-date">
              Published: {pub.published_date}
            </div>
            
            <div className="result-abstract">
              {highlightText(pub.abstract.substring(0, 300), query)}
              {pub.abstract.length > 300 && '...'}
            </div>
            
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

export default App
