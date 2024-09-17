const express = require('express');
const cors = require('cors');
const { spawn } = require('child_process');
const app = express();
const port = 3001;

app.use(cors());
app.use(express.json());

app.post('/rank-candidates', (req, res) => {
    const { requiredRole, topN, skillsWeight } = req.body;
    const mainDataFilePath = './skills_and_salaries_categories_p.csv';
    const jobRolesDataFilePath = './job_skills.csv';
    const testDataFilePath = './FinalResult.csv';

    console.log(`Received request for role: ${requiredRole}, topN: ${topN}, skillsWeight: ${skillsWeight}`);

    const pythonArgs = [
        '-u',
        'candidate_ranking.py',  // Change this line
        requiredRole,
        mainDataFilePath,
        jobRolesDataFilePath,
        testDataFilePath,
        '--top_n', topN.toString(),
        '--skills_weight', skillsWeight.toString()
    ];

    const pythonProcess = spawn('python', pythonArgs);

    let result = '';
    let error = '';

    pythonProcess.stdout.on('data', (data) => {
        result += data.toString();
        console.log("Partial output:", data.toString());
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
            // Try to parse the result
            const jsonStart = result.indexOf('[');
            const jsonEnd = result.lastIndexOf(']');
            if (jsonStart !== -1 && jsonEnd !== -1) {
                const jsonResult = JSON.parse(result.slice(jsonStart, jsonEnd + 1));
                res.json(jsonResult);
            } else {
                console.error("Invalid JSON format");
                res.status(400).json({ error: "Invalid JSON format", details: result });
            }
        } catch (e) {
            console.error(`Failed to parse result: ${e.message}`);
            res.status(500).json({ error: 'Failed to parse result', details: result });
        }
    });
});

app.listen(port, () => {
    console.log(`Server running on port ${port}`);
});
