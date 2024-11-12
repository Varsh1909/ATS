// import React, { useState } from 'react';
// import axios from 'axios';

// const CandidateRanking = () => {
//   const [requiredRole, setRequiredRole] = useState('');
//   const [topN, setTopN] = useState(5);
//   const [startDate, setStartDate] = useState('');
//   const [endDate, setEndDate] = useState('');
//   const [maxExperience, setMaxExperience] = useState('');
//   const [mandatorySkills, setMandatorySkills] = useState(['']);
//   const [optionalSkills, setOptionalSkills] = useState(['']);
//   const [results, setResults] = useState(null);
//   const [loading, setLoading] = useState(false);
//   const [error, setError] = useState(null);

//   const handleAddSkill = (skillType) => {
//     if (skillType === 'mandatory') {
//       setMandatorySkills([...mandatorySkills, '']);
//     } else {
//       setOptionalSkills([...optionalSkills, '']);
//     }
//   };

//   const handleSkillChange = (index, value, skillType) => {
//     if (skillType === 'mandatory') {
//       const newSkills = [...mandatorySkills];
//       newSkills[index] = value;
//       setMandatorySkills(newSkills);
//     } else {
//       const newSkills = [...optionalSkills];
//       newSkills[index] = value;
//       setOptionalSkills(newSkills);
//     }
//   };

//   const handleSubmit = async (e) => {
//     e.preventDefault();
//     setLoading(true);
//     setError(null);

//     const start = startDate ? new Date(startDate) : null;
//     const end = endDate ? new Date(endDate) : null;

//     if (startDate && isNaN(start)) {
//       setError("Invalid start date format. Please use YYYY-MM-DD.");
//       setLoading(false);
//       return;
//     }

//     if (endDate && isNaN(end)) {
//       setError("Invalid end date format. Please use YYYY-MM-DD.");
//       setLoading(false);
//       return;
//     }

//     if (start && end && start > end) {
//       setError("'Start Date' cannot be after 'End Date'.");
//       setLoading(false);
//       return;
//     }

//     try {
//       const response = await axios.post('http://localhost:3001/rank-candidates', {
//         requiredRole,
//         topN,
//         startDate: startDate || null,
//         endDate: endDate || null,
//         maxExperience: maxExperience ? parseInt(maxExperience) : null,
//         mandatorySkills: mandatorySkills.filter(skill => skill.trim() !== ''),
//         optionalSkills: optionalSkills.filter(skill => skill.trim() !== '')
//       });
//       setResults(response.data);
//     } catch (err) {
//       console.error('Error details:', err.response?.data || err.message);
//       setError(err.response?.data?.error || 'An error occurred while fetching results. Please check the console for more details.');
//     } finally {
//       setLoading(false);
//     }
//   };

//   return (
//     <div className="container mx-auto p-4">
//       <h1 className="text-2xl font-bold mb-4">Candidate Ranking</h1>
//       <form onSubmit={handleSubmit} className="mb-4">
//         <div className="mb-4">
//           <label htmlFor="requiredRole" className="block mb-2">Required Role:</label>
//           <input
//             type="text"
//             id="requiredRole"
//             value={requiredRole}
//             onChange={(e) => setRequiredRole(e.target.value)}
//             className="w-full p-2 border rounded"
//             required
//           />
//         </div>

//         <div className="mb-4">
//           <label htmlFor="topN" className="block mb-2">Number of Top Candidates:</label>
//           <input
//             type="number"
//             id="topN"
//             value={topN}
//             onChange={(e) => setTopN(parseInt(e.target.value))}
//             className="w-full p-2 border rounded"
//             required
//             min="1"
//           />
//         </div>

//         <div className="mb-4">
//           <label htmlFor="maxExperience" className="block mb-2">Maximum Experience (years):</label>
//           <input
//             type="number"
//             id="maxExperience"
//             value={maxExperience}
//             onChange={(e) => setMaxExperience(e.target.value)}
//             className="w-full p-2 border rounded"
//             min="0"
//           />
//         </div>

//         <div className="mb-4">
//           <label htmlFor="startDate" className="block mb-2">Start Date (optional):</label>
//           <input
//             type="date"
//             id="startDate"
//             value={startDate}
//             onChange={(e) => setStartDate(e.target.value)}
//             className="w-full p-2 border rounded"
//           />
//         </div>

//         <div className="mb-4">
//           <label htmlFor="endDate" className="block mb-2">End Date (optional):</label>
//           <input
//             type="date"
//             id="endDate"
//             value={endDate}
//             onChange={(e) => setEndDate(e.target.value)}
//             className="w-full p-2 border rounded"
//           />
//         </div>

//         <div className="mb-4">
//           <label className="block mb-2">Mandatory Skills:</label>
//           {mandatorySkills.map((skill, index) => (
//             <div key={index} className="flex mb-2">
//               <input
//                 type="text"
//                 value={skill}
//                 onChange={(e) => handleSkillChange(index, e.target.value, 'mandatory')}
//                 className="w-full p-2 border rounded mr-2"
//               />
//               {index === mandatorySkills.length - 1 && (
//                 <button
//                   type="button"
//                   onClick={() => handleAddSkill('mandatory')}
//                   className="bg-green-500 text-white px-4 py-2 rounded"
//                 >
//                   +
//                 </button>
//               )}
//             </div>
//           ))}
//         </div>

//         <div className="mb-4">
//           <label className="block mb-2">Optional Skills:</label>
//           {optionalSkills.map((skill, index) => (
//             <div key={index} className="flex mb-2">
//               <input
//                 type="text"
//                 value={skill}
//                 onChange={(e) => handleSkillChange(index, e.target.value, 'optional')}
//                 className="w-full p-2 border rounded mr-2"
//               />
//               {index === optionalSkills.length - 1 && (
//                 <button
//                   type="button"
//                   onClick={() => handleAddSkill('optional')}
//                   className="bg-green-500 text-white px-4 py-2 rounded"
//                 >
//                   +
//                 </button>
//               )}
//             </div>
//           ))}
//         </div>

//         <button
//           type="submit"
//           className="bg-blue-500 text-white px-4 py-2 rounded"
//           disabled={loading}
//         >
//           {loading ? 'Loading...' : 'Rank Candidates'}
//         </button>
//       </form>

//       {error && <p className="text-red-500 mb-4">{error}</p>}

//       {results && (
//         <div className="overflow-x-auto">
//           <h2 className="text-xl font-bold mb-2">Results:</h2>
//           <table className="w-full border-collapse border">
//             <thead>
//               <tr className="bg-gray-200">
//                 <th className="border p-2">Rank</th>
//                 <th className="border p-2">Candidate ID</th>
//                 <th className="border p-2">Full Name</th>
//                 <th className="border p-2">Email</th>
//                 <th className="border p-2">Status</th>
//                 <th className="border p-2">Experience</th>
//                 <th className="border p-2">Profile Title</th>
//                 <th className="border p-2">Source</th>
//                 <th className="border p-2">Date</th>
//                 <th className="border p-2">Predicted Salary</th>
//                 <th className="border p-2">Skills</th>
//                 <th className="border p-2">Has Mandatory Skills</th>
//                 <th className="border p-2">Has Optional Skills</th>
//               </tr>
//             </thead>
//             <tbody>
//               {results.map((candidate, index) => (
//                 <tr key={candidate.candidate_id} className={index % 2 === 0 ? 'bg-white' : 'bg-gray-100'}>
//                   <td className="border p-2">{index + 1}</td>
//                   <td className="border p-2">{candidate.candidate_id}</td>
//                   <td className="border p-2">{candidate.full_name}</td>
//                   <td className="border p-2">{candidate.email}</td>
//                   <td className="border p-2">{candidate.candidate_status}</td>
//                   <td className="border p-2">{candidate.experience}</td>
//                   <td className="border p-2">{candidate.profile_title}</td>
//                   <td className="border p-2">{candidate.source_name}</td>
//                   <td className="border p-2">{candidate.date}</td>
//                   <td className="border p-2">${candidate.Predicted_Salary.toFixed(2)}</td>
//                   <td className="border p-2">{candidate.skills}</td>
//                   <td className="border p-2">{candidate.has_mandatory_skills ? 'Yes' : 'No'}</td>
//                   <td className="border p-2">{candidate.has_optional_skills ? 'Yes' : 'No'}</td>
//                 </tr>
//               ))}
//             </tbody>
//           </table>
//         </div>
//       )}
//     </div>
//   );
// };

// export default CandidateRanking;

import React, { useState } from 'react';

const CandidateRanking = () => {
  const [requiredRole, setRequiredRole] = useState('');
  const [topN, setTopN] = useState(5);
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [maxExperience, setMaxExperience] = useState('');
  const [mandatorySkills, setMandatorySkills] = useState(['']);
  const [optionalSkills, setOptionalSkills] = useState(['']);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleAddSkill = (skillType) => {
    if (skillType === 'mandatory') {
      setMandatorySkills([...mandatorySkills, '']);
    } else {
      setOptionalSkills([...optionalSkills, '']);
    }
  };

  const handleSkillChange = (index, value, skillType) => {
    if (skillType === 'mandatory') {
      const newSkills = [...mandatorySkills];
      newSkills[index] = value;
      setMandatorySkills(newSkills);
    } else {
      const newSkills = [...optionalSkills];
      newSkills[index] = value;
      setOptionalSkills(newSkills);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

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
      const response = await fetch('https://backend-backend-hntl.onrender.com/rank-candidates', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          requiredRole,
          topN,
          startDate: startDate || null,
          endDate: endDate || null,
          maxExperience: maxExperience ? parseInt(maxExperience) : null,
          mandatorySkills: mandatorySkills.filter(skill => skill.trim() !== ''),
          optionalSkills: optionalSkills.filter(skill => skill.trim() !== '')
        })
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
        
        <div className="bg-white rounded-lg shadow-sm p-6 mb-8">
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label htmlFor="requiredRole" className="block text-sm font-medium text-gray-700 mb-2">
                  Required Role
                </label>
                <input
                  type="text"
                  id="requiredRole"
                  value={requiredRole}
                  onChange={(e) => setRequiredRole(e.target.value)}
                  className="w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                  required
                />
              </div>

              <div>
                <label htmlFor="topN" className="block text-sm font-medium text-gray-700 mb-2">
                  Number of Top Candidates
                </label>
                <input
                  type="number"
                  id="topN"
                  value={topN}
                  onChange={(e) => setTopN(parseInt(e.target.value))}
                  className="w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                  required
                  min="1"
                />
              </div>

              <div>
                <label htmlFor="maxExperience" className="block text-sm font-medium text-gray-700 mb-2">
                  Maximum Experience (years)
                </label>
                <input
                  type="number"
                  id="maxExperience"
                  value={maxExperience}
                  onChange={(e) => setMaxExperience(e.target.value)}
                  className="w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                  min="0"
                />
              </div>

              <div>
                <label htmlFor="startDate" className="block text-sm font-medium text-gray-700 mb-2">
                  Start Date (optional)
                </label>
                <input
                  type="date"
                  id="startDate"
                  value={startDate}
                  onChange={(e) => setStartDate(e.target.value)}
                  className="w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                />
              </div>

              <div>
                <label htmlFor="endDate" className="block text-sm font-medium text-gray-700 mb-2">
                  End Date (optional)
                </label>
                <input
                  type="date"
                  id="endDate"
                  value={endDate}
                  onChange={(e) => setEndDate(e.target.value)}
                  className="w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                />
              </div>
            </div>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Mandatory Skills
                </label>
                {mandatorySkills.map((skill, index) => (
                  <div key={index} className="flex gap-2 mb-2">
                    <input
                      type="text"
                      value={skill}
                      onChange={(e) => handleSkillChange(index, e.target.value, 'mandatory')}
                      className="flex-1 rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                    />
                    {index === mandatorySkills.length - 1 && (
                      <button
                        type="button"
                        onClick={() => handleAddSkill('mandatory')}
                        className="inline-flex items-center px-3 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500"
                      >
                        +
                      </button>
                    )}
                  </div>
                ))}
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Optional Skills
                </label>
                {optionalSkills.map((skill, index) => (
                  <div key={index} className="flex gap-2 mb-2">
                    <input
                      type="text"
                      value={skill}
                      onChange={(e) => handleSkillChange(index, e.target.value, 'optional')}
                      className="flex-1 rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                    />
                    {index === optionalSkills.length - 1 && (
                      <button
                        type="button"
                        onClick={() => handleAddSkill('optional')}
                        className="inline-flex items-center px-3 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500"
                      >
                        +
                      </button>
                    )}
                  </div>
                ))}
              </div>
            </div>

            <div className="flex justify-end">
              <button
                type="submit"
                className="inline-flex items-center px-4 py-2 border border-transparent text-base font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50"
                disabled={loading}
              >
                {loading ? 'Loading...' : 'Rank Candidates'}
              </button>
            </div>
          </form>
        </div>

        {error && (
          <div className="rounded-md bg-red-50 p-4 mb-8">
            <div className="flex">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="ml-3">
                <p className="text-sm font-medium text-red-800">{error}</p>
              </div>
            </div>
          </div>
        )}

        {results && (
          <div className="bg-white rounded-lg shadow-sm overflow-hidden">
            <div className="px-6 py-4 border-b border-gray-200">
              <h2 className="text-xl font-bold text-gray-900">Results</h2>
            </div>
            <div className="overflow-x-auto max-h-96" style={{ backgroundColor: 'rgb(249, 250, 251)' }}>
              <div className="inline-block min-w-full align-middle">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th scope="col" className="sticky top-0 z-10 bg-gray-50 px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Rank</th>
                      <th scope="col" className="sticky top-0 z-10 bg-gray-50 px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">ID</th>
                      <th scope="col" className="sticky top-0 z-10 bg-gray-50 px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Name</th>
                      <th scope="col" className="sticky top-0 z-10 bg-gray-50 px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Email</th>
                      <th scope="col" className="sticky top-0 z-10 bg-gray-50 px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Experience</th>
                      <th scope="col" className="sticky top-0 z-10 bg-gray-50 px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Title</th>
                      <th scope="col" className="sticky top-0 z-10 bg-gray-50 px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Source</th>
                      <th scope="col" className="sticky top-0 z-10 bg-gray-50 px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Date</th>
                      <th scope="col" className="sticky top-0 z-10 bg-gray-50 px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Salary</th>
                      <th scope="col" className="sticky top-0 z-10 bg-gray-50 px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Skills</th>
                      <th scope="col" className="sticky top-0 z-10 bg-gray-50 px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Mandatory</th>
                      <th scope="col" className="sticky top-0 z-10 bg-gray-50 px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Optional</th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {results.map((candidate, index) => (
                      <tr key={candidate.candidate_id} className={index % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{index + 1}</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{candidate.candidate_id}</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{candidate.full_name}</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{candidate.email}</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{candidate.experience}</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{candidate.profile_title}</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{candidate.source_name}</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{candidate.date}</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">${candidate.Predicted_Salary.toFixed(2)}</td>
                        <td className="px-6 py-4 whitespace-normal text-sm text-gray-500 max-w-xs">{candidate.skills}</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          <span className={`px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full ${
                            candidate.has_mandatory_skills 
                              ? 'bg-green-100 text-green-800' 
                              : 'bg-red-100 text-red-800'
                          }`}>
                            {candidate.has_mandatory_skills ? 'Yes' : 'No'}
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          <span className={`px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full ${
                            candidate.has_optional_skills 
                              ? 'bg-green-100 text-green-800' 
                              : 'bg-gray-100 text-gray-800'
                          }`}>
                            {candidate.has_optional_skills ? 'Yes' : 'No'}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default CandidateRanking;