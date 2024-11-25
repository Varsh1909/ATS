import React, { useState } from 'react';

const Form = ({ onSubmit, loading, error }) => {
  const [requiredRole, setRequiredRole] = useState('');
  const [topN, setTopN] = useState(5);
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [maxExperience, setMaxExperience] = useState('');
  const [minExperience, setMinExperience] = useState('');
  const [mandatorySkills, setMandatorySkills] = useState(['']);
  const [optionalSkills, setOptionalSkills] = useState(['']);

  const handleAddSkill = (skillType) => {
    if (skillType === 'mandatory') {
      setMandatorySkills([...mandatorySkills, '']);
    } else {
      setOptionalSkills([...optionalSkills, '']);
    }
  };

  const handleSkillChange = (index, value, skillType) => {
    const setSkills = skillType === 'mandatory' ? setMandatorySkills : setOptionalSkills;
    const skills = skillType === 'mandatory' ? mandatorySkills : optionalSkills;
    const newSkills = [...skills];
    newSkills[index] = value;
    setSkills(newSkills);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    
    const start = startDate ? new Date(startDate) : null;
    const end = endDate ? new Date(endDate) : null;

    // Input validation
    if (start && end && start > end) {
      alert("'Start Date' cannot be after 'End Date'.");
      return;
    }

    onSubmit({
      requiredRole,
      topN,
      startDate: startDate || null,
      endDate: endDate || null,
      maxExperience: maxExperience ? parseInt(maxExperience) : null,
      minExperience: minExperience ? parseInt(minExperience) : null,
      mandatorySkills: mandatorySkills.filter(skill => skill.trim() !== ''),
      optionalSkills: optionalSkills.filter(skill => skill.trim() !== '')
    });
  };

  return (
    <>
      <div className="bg-white rounded-lg shadow-md p-6 mb-8">
        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Required Role</label>
              <input
                type="text"
                value={requiredRole}
                onChange={(e) => setRequiredRole(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                required
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Top Candidates</label>
              <input
                type="number"
                value={topN}
                onChange={(e) => setTopN(parseInt(e.target.value))}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                required
                min="1"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Start Date</label>
              <input
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">End Date</label>
              <input
                type="date"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Min Experience</label>
              <input
                type="number"
                value={minExperience}
                onChange={(e) => setMinExperience(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                min="0"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Max Experience</label>
              <input
                type="number"
                value={maxExperience}
                onChange={(e) => setMaxExperience(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                min="0"
              />
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Mandatory Skills</label>
              {mandatorySkills.map((skill, index) => (
                <div key={index} className="flex mb-2">
                  <input
                    type="text"
                    value={skill}
                    onChange={(e) => handleSkillChange(index, e.target.value, 'mandatory')}
                    className="flex-1 px-3 py-2 border border-gray-300 rounded-l-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                  {index === mandatorySkills.length - 1 && (
                    <button
                      type="button"
                      onClick={() => handleAddSkill('mandatory')}
                      className="px-3 py-2 bg-green-500 text-white rounded-r-md hover:bg-green-600"
                    >
                      +
                    </button>
                  )}
                </div>
              ))}
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Optional Skills</label>
              {optionalSkills.map((skill, index) => (
                <div key={index} className="flex mb-2">
                  <input
                    type="text"
                    value={skill}
                    onChange={(e) => handleSkillChange(index, e.target.value, 'optional')}
                    className="flex-1 px-3 py-2 border border-gray-300 rounded-l-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                  {index === optionalSkills.length - 1 && (
                    <button
                      type="button"
                      onClick={() => handleAddSkill('optional')}
                      className="px-3 py-2 bg-green-500 text-white rounded-r-md hover:bg-green-600"
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
              className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
              disabled={loading}
            >
              {loading ? 'Loading...' : 'Rank Candidates'}
            </button>
          </div>
        </form>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-800 px-4 py-3 rounded relative" role="alert">
          <span className="block sm:inline">{error}</span>
        </div>
      )}
    </>
  );
};

export default Form;