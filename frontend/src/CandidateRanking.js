import React, { useState } from 'react';
import axios from 'axios';

const CandidateRanking = () => {
  const [requiredRole, setRequiredRole] = useState('');
  const [topN, setTopN] = useState(5);
  const [skillsWeight, setSkillsWeight] = useState(0.6);
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    // Check if the provided dates are valid
    const start = startDate ? new Date(startDate) : null;
    const end = endDate ? new Date(endDate) : null;

    if (startDate && isNaN(start)) {
      setError("Invalid start date format. Please use YYYY-MM-DD.");
      setLoading(false);
      return;
    }

    if (endDate && isNaN(end)) {
      setError("Invalid end date format. Please use YYYY-MM-DD.");
      setLoading(false);
      return;
    }

    if (start && end && start > end) {
      setError("'Start Date' cannot be after 'End Date'.");
      setLoading(false);
      return;
    }

    try {
      const response = await axios.post('http://localhost:3001/rank-candidates', {
        requiredRole,
        topN,
        skillsWeight,
        startDate: startDate || null, // Send null if startDate is empty
        endDate: endDate || null // Send null if endDate is empty
      });
      setResults(response.data);
    } catch (err) {
      console.error('Error details:', err.response?.data || err.message);
      setError(err.response?.data?.error || 'An error occurred while fetching results. Please check the console for more details.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">Candidate Ranking</h1>
      <form onSubmit={handleSubmit} className="mb-4">
        {/* Required Role Input */}
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

        {/* Number of Top Candidates Input */}
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

        {/* Skills Weight Input */}
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

        {/* Start Date Input */}
        <div className="mb-4">
          <label htmlFor="startDate" className="block mb-2">Start Date (optional):</label>
          <input
            type="date"
            id="startDate"
            value={startDate}
            onChange={(e) => setStartDate(e.target.value)}
            className="w-full p-2 border rounded"
          />
        </div>

        {/* End Date Input */}
        <div className="mb-4">
          <label htmlFor="endDate" className="block mb-2">End Date (optional):</label>
          <input
            type="date"
            id="endDate"
            value={endDate}
            onChange={(e) => setEndDate(e.target.value)}
            className="w-full p-2 border rounded"
          />
        </div>

        {/* Submit Button */}
        <button
          type="submit"
          className="bg-blue-500 text-white px-4 py-2 rounded"
          disabled={loading}
        >
          {loading ? 'Loading...' : 'Rank Candidates'}
        </button>
      </form>

      {/* Error Message */}
      {error && <p className="text-red-500 mb-4">{error}</p>}

      {/* Results Table */}
      {results && (
        <div className="overflow-x-auto">
          <h2 className="text-xl font-bold mb-2">Results:</h2>
          <table className="w-full border-collapse border">
            <thead>
              <tr className="bg-gray-200">
                <th className="border p-2">Rank</th>
                <th className="border p-2">Candidate ID</th>
                <th className="border p-2">Full Name</th>
                <th className="border p-2">Email</th>
                <th className="border p-2">Status</th>
                <th className="border p-2">Experience</th>
                <th className="border p-2">Profile Title</th>
                <th className="border p-2">Source</th>
                <th className="border p-2">Date</th>
                <th className="border p-2">Weighted Score</th>
                <th className="border p-2">Predicted Salary</th>
              </tr>
            </thead>
            <tbody>
              {results.map((candidate, index) => (
                <tr key={candidate.candidate_id} className={index % 2 === 0 ? 'bg-white' : 'bg-gray-100'}>
                  <td className="border p-2">{index + 1}</td>
                  <td className="border p-2">{candidate.candidate_id}</td>
                  <td className="border p-2">{candidate.full_name}</td>
                  <td className="border p-2">{candidate.email}</td>
                  <td className="border p-2">{candidate.candidate_status}</td>
                  <td className="border p-2">{candidate.experience}</td>
                  <td className="border p-2">{candidate.profile_title}</td>
                  <td className="border p-2">{candidate.source_name}</td>
                  <td className="border p-2">{candidate.date}</td>
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
