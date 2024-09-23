const express = require('express');
const cors = require('cors');
const { spawn } = require('child_process');
const app = express();
const port = 3001;

app.use(cors());
app.use(express.json());

app.post('/rank-candidates', (req, res) => {
  const { requiredRole, topN, skillsWeight } = req.body;
  const dataFilePath = 'E:\\ats\\FinalResult.csv';

  console.log(`Received request for role: ${requiredRole}, topN: ${topN}, skillsWeight: ${skillsWeight}`);

  const pythonProcess = spawn('python', ['candidate_ranking.py', requiredRole, topN.toString(), skillsWeight.toString(), dataFilePath]);

  let result = '';
  let error = '';

  pythonProcess.stdout.on('data', (data) => {
    result += data.toString();
  });

  pythonProcess.stderr.on('data', (data) => {
    error += data.toString();
    console.error(`Python script error: ${data}`);
  });

  pythonProcess.on('close', (code) => {
    console.log(`Python script exited with code ${code}`);
    if (code !== 0) {
      console.error(`Error in Python script: ${error}`);
      return res.status(500).json({ error: error });
    }
    try {
      // Extract the JSON part from the result
      const jsonStart = result.indexOf('[');
      const jsonEnd = result.lastIndexOf(']') + 1;
      const jsonResult = JSON.parse(result.slice(jsonStart, jsonEnd));
      res.json(jsonResult);
    } catch (e) {
      console.error(`Failed to parse result: ${e.message}`);
      res.status(500).json({ error: 'Failed to parse result', details: result });
    }
  });
});

app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});