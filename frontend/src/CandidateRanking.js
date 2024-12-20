import React, { useState } from 'react';
import axios from 'axios';

const CandidateRanking = () => {
  const [requiredRole, setRequiredRole] = useState('');
  const [topN, setTopN] = useState(5);
  const [skillsWeight, setSkillsWeight] = useState(0.6);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      //Updated the link for prod to link 
      const response = await axios.post('https://backend-backend-hntl.onrender.com/rank-candidates', {
        requiredRole,
        topN,
        skillsWeight
      });
      setResults(response.data);
    } catch (err) {
      console.error('Error details:', err.response?.data || err.message);
      setError('An error occurred while fetching results. Please check the console for more details.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">Candidate Ranking</h1>
      <form onSubmit={handleSubmit} className="mb-4">
        <div className="mb-4">
          <label htmlFor="requiredRole" className="block mb-2">Required Role:</label>
          <input
            type="text"
            id="requiredRole"
            value={requiredRole}
            onChange={(e) => setRequiredRole(e.target.value)}
            className="w-full p-2 border rounded"
            required
          />
        </div>
        <div className="mb-4">
          <label htmlFor="topN" className="block mb-2">Number of Top Candidates:</label>
          <input
            type="number"
            id="topN"
            value={topN}
            onChange={(e) => setTopN(parseInt(e.target.value))}
            className="w-full p-2 border rounded"
            required
            min="1"
          />
        </div>
        <div className="mb-4">
          <label htmlFor="skillsWeight" className="block mb-2">Skills Weight (0.1 to 1.0):</label>
          <input
            type="number"
            id="skillsWeight"
            value={skillsWeight}
            onChange={(e) => setSkillsWeight(parseFloat(e.target.value))}
            className="w-full p-2 border rounded"
            required
            min="0.1"
            max="1.0"
            step="0.1"
          />
        </div>
        <button type="submit" className="bg-blue-500 text-white px-4 py-2 rounded" disabled={loading}>
          {loading ? 'Loading...' : 'Rank Candidates'}
        </button>
      </form>

      {error && <p className="text-red-500">{error}</p>}

      {results && (
        <div className="overflow-x-auto">
          <h2 className="text-xl font-bold mb-2">Results:</h2>
          <table className="w-full border-collapse border">
            <thead>
              <tr className="bg-gray-200">
                <th className="border p-2">Rank</th>
                <th className="border p-2">Candidate ID</th>
                <th className="border p-2">Full Name</th>
                <th className="border p-2">Experience</th>
                <th className="border p-2">Profile Title</th>
                <th className="border p-2">Weighted Score</th>
                <th className="border p-2">Predicted Salary</th>
              </tr>
            </thead>
            <tbody>
              {results.map((candidate, index) => (
                <tr key={candidate.candidate_id}>
                  <td className="border p-2">{index + 1}</td>
                  <td className="border p-2">{candidate.candidate_id}</td>
                  <td className="border p-2">{candidate.full_name}</td>
                  <td className="border p-2">{candidate.experience}</td>
                  <td className="border p-2">{candidate.profile_title}</td>
                  <td className="border p-2">{candidate.Weighted_Score.toFixed(2)}</td>
                  <td className="border p-2">${candidate.Predicted_Salary.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default CandidateRanking;
