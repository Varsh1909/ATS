import React, { useState } from 'react';
import Form from './Form';
import ResultsTable from './ResultsTable';

const CandidateRanking = () => {
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [darkMode, setDarkMode] = useState(false);

  const handleSubmit = async (formData) => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('https://backend-backend-hntl.onrender.com/rank-candidates', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData)
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'An error occurred while fetching results.');
      }
      
      const data = await response.json();
      setResults(data);
    } catch (err) {
      console.error('Error details:', err);
      setError(err.message || 'An error occurred while fetching results.');
    } finally {
      setLoading(false);
    }
  };

  const toggleDarkMode = () => {
    setDarkMode(!darkMode);
  };

  return (
    <div className={`App ${darkMode ? 'dark-mode' : 'light-mode'}`}>
      <div className="main-container">
        <header className="App-header">
          <h1>Candidate Ranking System</h1>
          {/* <button className="toggle-btn" onClick={toggleDarkMode}>
            {darkMode ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
          </button> */}
        </header>

        <div className="content-wrapper">
          {/* Left side - Form */}
          <div className="form-section">
            <Form onSubmit={handleSubmit} loading={loading} error={error} />
          </div>

          {/* Right side - Results */}
          <div className="results-section">
            <div className="search-container">
              <input
                type="text"
                placeholder="Search candidates..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="search-input"
              />
            </div>
            
            <div className="table-wrapper">
              {loading ? (
                <div className="loading">Loading...</div>
              ) : error ? (
                <div className="error">{error}</div>
              ) : results ? (
                <ResultsTable 
                  results={results.filter(candidate => 
                    candidate.name?.toLowerCase().includes(searchTerm.toLowerCase()) ||
                    candidate.email?.toLowerCase().includes(searchTerm.toLowerCase()) ||
                    candidate.id?.toString().includes(searchTerm)
                  )} 
                />
              ) : (
                <div className="no-results">
                  Submit the form to see candidate rankings
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CandidateRanking;