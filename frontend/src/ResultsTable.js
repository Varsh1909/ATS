// import React, { useState, useRef, useEffect } from 'react';
// import { ArrowUpDown, Download, Search, Eye, EyeOff } from 'lucide-react';

// export default function ResultsTable({ results = [] }) {
//   const [searchTerm, setSearchTerm] = useState("");
//   const [sortConfig, setSortConfig] = useState({ key: "", direction: "asc" });
//   const [columnVisibility, setColumnVisibility] = useState({});
//   const tableRef = useRef(null);

//   const columns = [
//     { key: "rank", label: "Rank" },
//     { key: "id", label: "ID" },
//     { key: "full_name", label: "Name" },
//     { key: "email", label: "Email" },
//     { key: "experience", label: "Experience" },
//     { key: "profile_title", label: "Title" },
//     { key: "source_name", label: "Source" },
//     { key: "date", label: "Date" },
//     { key: "Predicted_Salary", label: "Salary" },
//     { key: "skills", label: "Skills" },
//     { key: "has_mandatory_skills", label: "Mandatory Skills" },
//     { key: "has_optional_skills", label: "Optional Skills" },
//     { key: "work_authorization", label: "Work Authorization" },
//     { key: "city", label: "City" },
//     { key: "state", label: "State" },
//     { key: "willing_to_relocate", label: "Relocate" },
//   ];

//   useEffect(() => {
//     const initialColumnVisibility = {};
//     columns.forEach(column => {
//       initialColumnVisibility[column.key] = true;
//     });
//     setColumnVisibility(initialColumnVisibility);
//   }, []);

//   const filteredResults = results.filter((candidate) =>
//     Object.values(candidate).some(
//       (value) =>
//         value &&
//         value.toString().toLowerCase().includes(searchTerm.toLowerCase())
//     )
//   );

//   const sortedResults = [...filteredResults].sort((a, b) => {
//     if (!sortConfig.key) return 0;
//     const aValue = a[sortConfig.key];
//     const bValue = b[sortConfig.key];
//     if (aValue < bValue) return sortConfig.direction === "asc" ? -1 : 1;
//     if (aValue > bValue) return sortConfig.direction === "asc" ? 1 : -1;
//     return 0;
//   });

//   const exportToCSV = () => {
//     const visibleColumns = columns.filter(column => columnVisibility[column.key]);
//     const headers = visibleColumns.map(column => column.label);
    
//     const csvContent = [
//       headers.join(','),
//       ...sortedResults.map((candidate, index) => 
//         visibleColumns.map(column => {
//           if (column.key === 'rank') return index + 1;
//           let value = candidate[column.key];
//           if (typeof value === 'boolean') value = value ? 'Yes' : 'No';
//           return `"${value}"`;
//         }).join(',')
//       )
//     ].join('\n');

//     const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
//     const link = document.createElement('a');
//     link.href = URL.createObjectURL(blob);
//     link.download = 'candidates.csv';
//     link.click();
//   };

//   const handleSort = (key) => {
//     setSortConfig({
//       key,
//       direction:
//         sortConfig.key === key && sortConfig.direction === "asc" ? "desc" : "asc",
//     });
//   };

//   const toggleColumnVisibility = (key) => {
//     setColumnVisibility(prev => ({ ...prev, [key]: !prev[key] }));
//   };

//   return (
//     <div className="results-table-container">
//       <div className="table-controls">
//         <div className="search-container">
//           <Search className="search-icon" />
//           <input
//             type="text"
//             placeholder="Search candidates..."
//             value={searchTerm}
//             onChange={(e) => setSearchTerm(e.target.value)}
//             className="search-input"
//           />
//         </div>
//         <button onClick={exportToCSV} className="export-button">
//           <Download className="download-icon" />
//           Export to CSV
//         </button>
//       </div>

//       <div className="table-wrapper">
//         <table ref={tableRef} className="results-table">
//           <thead>
//             <tr>
//               {columns.map(({ key, label }) => (
//                 columnVisibility[key] && (
//                   <th key={key} className="table-header">
//                     <div className="header-content">
//                       <button onClick={() => handleSort(key)} className="sort-button">
//                         {label}
//                         <ArrowUpDown className="sort-icon" />
//                       </button>
//                       <button onClick={() => toggleColumnVisibility(key)} className="visibility-button">
//                         {columnVisibility[key] ? (
//                           <Eye className="eye-icon" />
//                         ) : (
//                           <EyeOff className="eye-off-icon" />
//                         )}
//                       </button>
//                     </div>
//                   </th>
//                 )
//               ))}
//             </tr>
//           </thead>
//           <tbody>
//             {sortedResults.map((candidate, index) => (
//               <tr key={candidate.id} className="table-row">
//                 {columns.map(({ key }) => (
//                   columnVisibility[key] && (
//                     <td key={key} className="table-cell">
//                       {key === 'rank' ? index + 1 : 
//                        key === 'has_mandatory_skills' || key === 'has_optional_skills' ? (
//                         <span className={`skill-badge ${candidate[key] ? "skill-yes" : "skill-no"}`}>
//                           {candidate[key] ? "Yes" : "No"}
//                         </span>
//                       ) : key === 'Predicted_Salary' ? (
//                         `$${candidate[key].toFixed(2)}`
//                       ) : (
//                         candidate[key]
//                       )}
//                     </td>
//                   )
//                 ))}
//               </tr>
//             ))}
//           </tbody>
//         </table>
//       </div>
//       <style jsx>{`
//         .results-table-container {
//           height: 100vh;
//           display: flex;
//           flex-direction: column;
//           gap: 1rem;
//           padding: 1rem;
//         }
//         .table-controls {
//           display: flex;
//           justify-content: space-between;
//           align-items: center;
//         }
//         .search-container {
//           display: flex;
//           align-items: center;
//           gap: 0.5rem;
//         }
//         .search-icon, .download-icon {
//           width: 1rem;
//           height: 1rem;
//         }
//         .search-input {
//           width: 300px;
//           padding: 0.5rem;
//           border: 1px solid #ccc;
//           border-radius: 4px;
//         }
//         .export-button {
//           display: flex;
//           align-items: center;
//           gap: 0.5rem;
//           padding: 0.5rem 1rem;
//           background-color: #3b82f6;
//           color: white;
//           border: none;
//           border-radius: 4px;
//           cursor: pointer;
//         }
//         .export-button:hover {
//           background-color: #2563eb;
//         }
//         .table-wrapper {
//           flex-grow: 1;
//           overflow: auto;
//           border: 1px solid #e5e7eb;
//           border-radius: 0.5rem;
//         }
//         .results-table {
//           width: 100%;
//           border-collapse: collapse;
//           font-size: 0.875rem;
//         }
//         .table-header {
//           position: sticky;
//           top: 0;
//           background-color: #f3f4f6;
//           z-index: 10;
//           padding: 0.75rem 1rem;
//           text-align: left;
//           font-weight: 600;
//         }
//         .header-content {
//           display: flex;
//           justify-content: space-between;
//           align-items: center;
//         }
//         .sort-button, .visibility-button {
//           background: none;
//           border: none;
//           cursor: pointer;
//           display: flex;
//           align-items: center;
//           gap: 0.25rem;
//         }
//         .sort-icon, .eye-icon, .eye-off-icon {
//           width: 1rem;
//           height: 1rem;
//         }
//         .table-row:nth-child(even) {
//           background-color: #f9fafb;
//         }
//         .table-row:hover {
//           background-color: #f3f4f6;
//         }
//         .table-cell {
//           padding: 0.75rem 1rem;
//           white-space: nowrap;
//           overflow: hidden;
//           text-overflow: ellipsis;
//         }
//         .skill-badge {
//           padding: 0.25rem 0.5rem;
//           border-radius: 9999px;
//           font-size: 0.75rem;
//         }
//         .skill-yes {
//           background-color: #d1fae5;
//           color: #065f46;
//         }
//         .skill-no {
//           background-color: #fee2e2;
//           color: #991b1b;
//         }
//       `}</style>
//     </div>
//   );
// }

// import React, { useState, useRef, useEffect } from 'react';
// import { ArrowUpDown, Download, Search, Eye, EyeOff } from 'lucide-react';

// export default function ResultsTable({ results = [] }) {
//   const [searchTerm, setSearchTerm] = useState("");
//   const [sortConfig, setSortConfig] = useState({ key: "", direction: "asc" });
//   const [columnVisibility, setColumnVisibility] = useState({});
//   const tableRef = useRef(null);

//   const columns = [
//     { key: "rank", label: "Rank" },
//     { key: "id", label: "ID" },
//     { key: "full_name", label: "Name" },
//     { key: "email", label: "Email" },
//     { key: "experience", label: "Experience" },
//     { key: "profile_title", label: "Title" },
//     { key: "source_name", label: "Source" },
//     { key: "date", label: "Date" },
//     { key: "Predicted_Salary", label: "Salary" },
//     { key: "skills", label: "Skills" },
//     { key: "has_mandatory_skills", label: "Mandatory Skills" },
//     { key: "has_optional_skills", label: "Optional Skills" },
//     { key: "work_authorization", label: "Work Authorization" },
//     { key: "city", label: "City" },
//     { key: "state", label: "State" },
//     { key: "willing_to_relocate", label: "Relocate" },
//   ];

//   useEffect(() => {
//     const initialColumnVisibility = {};
//     columns.forEach(column => {
//       initialColumnVisibility[column.key] = true;
//     });
//     setColumnVisibility(initialColumnVisibility);
//   }, []);

//   const filteredResults = results.filter((candidate) =>
//     Object.values(candidate).some(
//       (value) =>
//         value &&
//         value.toString().toLowerCase().includes(searchTerm.toLowerCase())
//     )
//   );

//   const sortedResults = [...filteredResults].sort((a, b) => {
//     if (!sortConfig.key) return 0;
//     const aValue = a[sortConfig.key];
//     const bValue = b[sortConfig.key];
//     if (aValue < bValue) return sortConfig.direction === "asc" ? -1 : 1;
//     if (aValue > bValue) return sortConfig.direction === "asc" ? 1 : -1;
//     return 0;
//   });

//   const exportToCSV = () => {
//     const visibleColumns = columns.filter(column => columnVisibility[column.key]);
//     const headers = visibleColumns.map(column => column.label);
    
//     const csvContent = [
//       headers.join(','),
//       ...sortedResults.map((candidate, index) => 
//         visibleColumns.map(column => {
//           if (column.key === 'rank') return index + 1;
//           let value = candidate[column.key];
//           if (typeof value === 'boolean') value = value ? 'Yes' : 'No';
//           return `"${value}"`;
//         }).join(',')
//       )
//     ].join('\n');

//     const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
//     const link = document.createElement('a');
//     link.href = URL.createObjectURL(blob);
//     link.download = 'candidates.csv';
//     link.click();
//   };

//   const handleSort = (key) => {
//     setSortConfig({
//       key,
//       direction:
//         sortConfig.key === key && sortConfig.direction === "asc" ? "desc" : "asc",
//     });
//   };

//   const toggleColumnVisibility = (key) => {
//     setColumnVisibility(prev => ({ ...prev, [key]: !prev[key] }));
//   };

//   return (
//     <div className="results-table-container">
//       <div className="table-controls">
//         <div className="search-container">
//           <Search className="search-icon" />
//           <input
//             type="text"
//             placeholder="Search candidates..."
//             value={searchTerm}
//             onChange={(e) => setSearchTerm(e.target.value)}
//             className="search-input"
//           />
//         </div>
//         <button onClick={exportToCSV} className="export-button">
//           <Download className="download-icon" />
//           Export to CSV
//         </button>
//       </div>

//       <div className="table-wrapper">
//         <div className="table-scrollable">
//           <table ref={tableRef} className="results-table">
//             <thead>
//               <tr>
//                 {columns.map(({ key, label }) => (
//                   columnVisibility[key] && (
//                     <th key={key} className="table-header">
//                       <div className="header-content">
//                         <button onClick={() => handleSort(key)} className="sort-button">
//                           {label}
//                           <ArrowUpDown className="sort-icon" />
//                         </button>
//                         <button onClick={() => toggleColumnVisibility(key)} className="visibility-button">
//                           {columnVisibility[key] ? (
//                             <Eye className="eye-icon" />
//                           ) : (
//                             <EyeOff className="eye-off-icon" />
//                           )}
//                         </button>
//                       </div>
//                     </th>
//                   )
//                 ))}
//               </tr>
//             </thead>
//             <tbody>
//               {sortedResults.map((candidate, index) => (
//                 <tr key={candidate.id} className="table-row">
//                   {columns.map(({ key }) => (
//                     columnVisibility[key] && (
//                       <td key={key} className="table-cell">
//                         {key === 'rank' ? index + 1 : 
//                         key === 'has_mandatory_skills' || key === 'has_optional_skills' ? (
//                           <span className={`skill-badge ${candidate[key] ? "skill-yes" : "skill-no"}`}>
//                             {candidate[key] ? "Yes" : "No"}
//                           </span>
//                         ) : key === 'Predicted_Salary' ? (
//                           `$${candidate[key].toFixed(2)}`
//                         ) : (
//                           candidate[key]
//                         )}
//                       </td>
//                     )
//                   ))}
//                 </tr>
//               ))}
//             </tbody>
//           </table>
//         </div>
//       </div>
//       <style jsx>{`
//         .results-table-container {
//           height: 100vh;
//           display: flex;
//           flex-direction: column;
//           gap: 1rem;
//           padding: 1rem;
//         }
//         .table-controls {
//           display: flex;
//           justify-content: space-between;
//           align-items: center;
//         }
//         .search-container {
//           display: flex;
//           align-items: center;
//           gap: 0.5rem;
//         }
//         .search-icon, .download-icon {
//           width: 1rem;
//           height: 1rem;
//         }
//         .search-input {
//           width: 300px;
//           padding: 0.5rem;
//           border: 1px solid #ccc;
//           border-radius: 4px;
//         }
//         .export-button {
//           display: flex;
//           align-items: center;
//           gap: 0.5rem;
//           padding: 0.5rem 1rem;
//           background-color: #3b82f6;
//           color: white;
//           border: none;
//           border-radius: 4px;
//           cursor: pointer;
//         }
//         .export-button:hover {
//           background-color: #2563eb;
//         }
//         .table-wrapper {
//           flex-grow: 1;
//           border: 1px solid #e5e7eb;
//           border-radius: 0.5rem;
//           overflow: hidden;
//           padding-left: 27%;
//         }
//         .table-scrollable {
//           overflow-x: hidden;
//         }
//         .results-table {
//           width: 85%;
//           border-collapse: collapse;
//           font-size: 0.875rem;
//         }
//         .table-header {
//           position: sticky;
//           top: 0;
//           background-color: #f3f4f6;
//           z-index: 10;
//           padding: 0.75rem 1rem;
//           text-align: left;
//           font-weight: 600;
//         }
//         .header-content {
//           display: flex;
//           justify-content: space-between;
//           align-items: center;
//         }
//         .sort-button, .visibility-button {
//           background: none;
//           border: none;
//           cursor: pointer;
//           display: flex;
//           align-items: center;
//           gap: 0.25rem;
//         }
//         .sort-icon, .eye-icon, .eye-off-icon {
//           width: 1rem;
//           height: 1rem;
//         }
//         .table-row:nth-child(even) {
//           background-color: #f9fafb;
//         }
//         .table-row:hover {
//           background-color: #f3f4f6;
//         }
//         .table-cell {
//           padding: 0.75rem 1rem;
//           white-space: nowrap;
//           overflow: hidden;
//           text-overflow: ellipsis;
//         }
//         .skill-badge {
//           padding: 0.25rem 0.5rem;
//           border-radius: 9999px;
//           font-size: 0.75rem;
//         }
//         .skill-yes {
//           background-color: #d1fae5;
//           color: #065f46;
//         }
//         .skill-no {
//           background-color: #fee2e2;
//           color: #991b1b;
//         }
//       `}</style>
//     </div>
//   );
// }

import React, { useState, useRef, useEffect } from 'react';
import { ArrowUpDown, Download, Search, Eye, EyeOff } from 'lucide-react';

export default function ResultsTable({ results = [] }) {
  const [searchTerm, setSearchTerm] = useState("");
  const [sortConfig, setSortConfig] = useState({ key: "", direction: "asc" });
  const [columnVisibility, setColumnVisibility] = useState({});
  const tableRef = useRef(null);

  const columns = [
    { key: "rank", label: "Rank" },
    { key: "id", label: "ID" },
    { key: "full_name", label: "Name" },
    { key: "email", label: "Email" },
    { key: "experience", label: "Experience" },
    { key: "profile_title", label: "Title" },
    { key: "source_name", label: "Source" },
    { key: "date", label: "Date" },
    { key: "Predicted_Salary", label: "Salary" },
    { key: "skills", label: "Skills" },
    { key: "has_mandatory_skills", label: "Mandatory Skills" },
    { key: "has_optional_skills", label: "Optional Skills" },
    { key: "work_authorization", label: "Work Authorization" },
    { key: "city", label: "City" },
    { key: "state", label: "State" },
    { key: "willing_to_relocate", label: "Relocate" },
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

  const exportToCSV = () => {
    const visibleColumns = columns.filter(column => columnVisibility[column.key]);
    const headers = visibleColumns.map(column => column.label);
    
    const csvContent = [
      headers.join(','),
      ...sortedResults.map((candidate, index) => 
        visibleColumns.map(column => {
          if (column.key === 'rank') return index + 1;
          let value = candidate[column.key];
          if (typeof value === 'boolean') value = value ? 'Yes' : 'No';
          return `"${value}"`;
        }).join(',')
      )
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'candidates.csv';
    link.click();
  };

  const handleSort = (key) => {
    setSortConfig({
      key,
      direction:
        sortConfig.key === key && sortConfig.direction === "asc" ? "desc" : "asc",
    });
  };

  const toggleColumnVisibility = (key) => {
    setColumnVisibility(prev => ({ ...prev, [key]: !prev[key] }));
  };

  return (
    <div className="h-screen flex flex-col p-4 gap-4">
      <div className="flex justify-between items-center">
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
        <button 
          onClick={exportToCSV} 
          className="flex items-center gap-2 px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600"
        >
          <Download className="w-4 h-4" />
          Export to CSV
        </button>
      </div>

      <div className="flex-1 border rounded-lg overflow-hidden">
        <div className="h-full overflow-auto">
          <div className="inline-block min-w-full align-middle">
            <div className="overflow-hidden">
              <table className="min-w-full">
                <thead className="bg-gray-50">
                  <tr>
                    {columns.map(({ key, label }) => (
                      columnVisibility[key] && (
                        <th 
                          key={key} 
                          className="sticky top-0 z-10 border-b border-gray-200 bg-gray-50 px-4 py-3 text-left text-sm font-semibold text-gray-900"
                        >
                          <div className="flex justify-between items-center">
                            <button 
                              onClick={() => handleSort(key)} 
                              className="flex items-center gap-1 hover:text-blue-600"
                            >
                              {label}
                              <ArrowUpDown className="w-4 h-4" />
                            </button>
                            <button 
                              onClick={() => toggleColumnVisibility(key)}
                              className="text-gray-500 hover:text-gray-700"
                            >
                              {columnVisibility[key] ? (
                                <Eye className="w-4 h-4" />
                              ) : (
                                <EyeOff className="w-4 h-4" />
                              )}
                            </button>
                          </div>
                        </th>
                      )
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200 bg-white">
                  {sortedResults.map((candidate, index) => (
                    <tr 
                      key={candidate.id}
                      className="hover:bg-gray-50"
                    >
                      {columns.map(({ key }) => (
                        columnVisibility[key] && (
                          <td 
                            key={key} 
                            className="whitespace-nowrap px-4 py-3 text-sm text-gray-900"
                          >
                            {key === 'rank' ? index + 1 : 
                             key === 'has_mandatory_skills' || key === 'has_optional_skills' ? (
                              <span className={`inline-flex rounded-full px-2 py-1 text-xs font-semibold ${
                                candidate[key] 
                                  ? "bg-green-100 text-green-800" 
                                  : "bg-red-100 text-red-800"
                              }`}>
                                {candidate[key] ? "Yes" : "No"}
                              </span>
                            ) : key === 'Predicted_Salary' ? (
                              `$${candidate[key].toFixed(2)}`
                            ) : (
                              candidate[key]
                            )}
                          </td>
                        )
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}