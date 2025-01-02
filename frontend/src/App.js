import React, { useState } from 'react';
import CandidateRanking from './CandidateRanking';
import './App.css';

const App = () => {
  const [darkMode, setDarkMode] = useState(true);

  const toggleMode = () => {
    setDarkMode(!darkMode);
  };

  return (
    <div className={darkMode ? 'App dark-mode' : 'App light-mode'}>
      <header className="App-header">
        <CandidateRanking />
      </header>
    </div>
  );
};

export default App;
