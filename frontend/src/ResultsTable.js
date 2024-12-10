import React, { useState, useRef, useEffect } from 'react';
import { ArrowUpDown, Search, CheckCircle2, XCircle, X } from 'lucide-react';

// Candidate Details Card Component

const CandidateDetailsCard = ({ candidate, onClose }) => {
  return (
    <div className="fixed inset-0 bg-gradient-to-br from-blue-100 to-blue-200 z-50 flex justify-center items-center p-4">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-5xl overflow-hidden relative border-4 border-blue-500">
      <h2 className="text-4xl font-extrabold mb-6 border-b-4 border-white pb-4">
            Candidate
            <br />
            Profile
          </h2>
        <button 
          onClick={onClose} 
          className="absolute top-4 right-4 text-gray-500 hover:text-gray-900 transition-colors"
        >
          <X className="w-8 h-8" />
        </button>

        <div className="p-8">
          <table>
            <tbody>
              <tr>
                <td className="font-semibold text-gray-600 p-3 bg-gray-50">Full Name</td>
                <td className="p-3">{candidate.full_name}</td>
              </tr>
              <tr>
                <td className="font-semibold text-gray-600 p-3 bg-gray-50">Email</td>
                <td className="p-3">{candidate.email}</td>
              </tr>
              <tr>
                <td className="font-semibold text-gray-600 p-3 bg-gray-50">Location</td>
                <td className="p-3">{candidate.city}, {candidate.state}</td>
              </tr>
              <tr>
                <td className="font-semibold text-gray-600 p-3 bg-gray-50">Profile Title</td>
                <td className="p-3">{candidate.profile_title}</td>
              </tr>
              <tr>
                <td className="font-semibold text-gray-600 p-3 bg-gray-50">Experience</td>
                <td className="p-3">{candidate.experience} years</td>
              </tr>
              <tr>
                <td className="font-semibold text-gray-600 p-3 bg-gray-50">Predicted Salary</td>
                <td className="p-3">${candidate.Predicted_Salary.toFixed(2)}</td>
              </tr>
              <tr>
                <td className="font-semibold text-gray-600 p-3 bg-gray-50">Work Authorization</td>
                <td className="p-3">{candidate.work_authorization}</td>
              </tr>
              <tr>
                <td className="font-semibold text-gray-600 p-3 bg-gray-50">Willing to Relocate</td>
                <td className="p-3">{candidate.willing_to_relocate}</td>
              </tr>
              <tr>
                <td className="font-semibold text-gray-600 p-3 bg-gray-50">Source</td>
                <td className="p-3">{candidate.source_name}</td>
              </tr>
              <tr>
                <td className="font-semibold text-gray-600 p-3 bg-gray-50">Skills</td>
                <td className="p-3">{candidate.skills}</td>
              </tr>
              <tr>
                <td className="font-semibold text-gray-600 p-3 bg-gray-50">Mandatory Skills</td>
                <td className="p-3">
                  {candidate.has_mandatory_skills ? (
                    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                      <CheckCircle2 className="w-4 h-4 mr-1" /> Yes
                    </span>
                  ) : (
                    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800">
                      <XCircle className="w-4 h-4 mr-1" /> No
                    </span>
                  )}
                </td>
              </tr>
              <tr>
                <td className="font-semibold text-gray-600 p-3 bg-gray-50">Optional Skills</td>
                <td className="p-3">
                  <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                    candidate.has_optional_skills 
                      ? "bg-green-100 text-green-800" 
                      : "bg-gray-100 text-gray-800"
                  }`}>
                    {candidate.has_optional_skills ? "Yes" : "No"}
                  </span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default function ResultsTable({ results = [] }) {
  const [searchTerm, setSearchTerm] = useState("");
  const [sortConfig, setSortConfig] = useState({ key: "", direction: "asc" });
  const [columnVisibility, setColumnVisibility] = useState({});
  const [selectedCandidate, setSelectedCandidate] = useState(null);

  const columns = [
    { key: "id", label: "ID" },
    { key: "full_name", label: "Name" },
    { key: "email", label: "Email" }
  ];

  useEffect(() => {
    const initialColumnVisibility = {};
    columns.forEach(column => {
      initialColumnVisibility[column.key] = true;
    });
    setColumnVisibility(initialColumnVisibility);
  }, []);

  const filteredResults = results.filter((candidate) =>
    Object.values(candidate).some(
      (value) =>
        value &&
        value.toString().toLowerCase().includes(searchTerm.toLowerCase())
    )
  );

  const sortedResults = [...filteredResults].sort((a, b) => {
    if (!sortConfig.key) return 0;
    const aValue = a[sortConfig.key];
    const bValue = b[sortConfig.key];
    if (aValue < bValue) return sortConfig.direction === "asc" ? -1 : 1;
    if (aValue > bValue) return sortConfig.direction === "asc" ? 1 : -1;
    return 0;
  });

  const handleSort = (key) => {
    setSortConfig({
      key,
      direction:
        sortConfig.key === key && sortConfig.direction === "asc" ? "desc" : "asc",
    });
  };

  const handleCandidateClick = (candidate) => {
    setSelectedCandidate(candidate);
  };

  return (
    <div className="h-screen flex flex-col p-4 gap-4">
      {/* Search Input */}
      <div className="flex items-center gap-2">
        <Search className="w-4 h-4 text-gray-500" />
        <input
          type="text"
          placeholder="Search candidates..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="w-80 p-2 border rounded-md"
        />
      </div>

      {/* Candidates Table */}
      <div className="flex-1 border rounded-lg overflow-hidden">
        <div className="h-full overflow-auto">
          <table className="min-w-full">
            <thead className="bg-gray-50">
              <tr>
                {columns.map(({ key, label }) => (
                  columnVisibility[key] && (
                    <th 
                      key={key} 
                      className="sticky top-0 z-10 border-b border-gray-200 bg-gray-50 px-4 py-3 text-left text-sm font-semibold text-gray-900"
                    >
                      <button 
                        onClick={() => handleSort(key)} 
                        className="flex items-center gap-1 hover:text-blue-600"
                      >
                        {label}
                        <ArrowUpDown className="w-4 h-4" />
                      </button>
                    </th>
                  )
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200 bg-white">
              {sortedResults.map((candidate) => (
                <tr 
                  key={candidate.candidate_id} 
                  onClick={() => handleCandidateClick(candidate)}
                  className="hover:bg-gray-50 cursor-pointer"
                >
                  {columns.map(({ key }) => (
                    columnVisibility[key] && (
                      <td 
                        key={key} 
                        className="whitespace-nowrap px-4 py-3 text-sm text-gray-900"
                      >
                        {candidate[key]}
                      </td>
                    )
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Candidate Details Card */}
      {selectedCandidate && (
        <CandidateDetailsCard 
          candidate={selectedCandidate} 
          onClose={() => setSelectedCandidate(null)}
        />
      )}
    </div>
  );
}