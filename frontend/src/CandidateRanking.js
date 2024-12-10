import React, { useState } from 'react';
import Form from './Form';
import ResultsTable from './ResultsTable';

const CandidateRanking = () => {
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

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

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-8">Candidate Ranking</h1>
        
        <Form 
          onSubmit={handleSubmit} 
          loading={loading} 
          error={error}
        />

        {results && (
          <ResultsTable results={results} />
        )}
      </div>
    </div>
  );
};

export default CandidateRanking;