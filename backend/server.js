// const express = require('express');
// const cors = require('cors');
// const { spawn } = require('child_process');
// const app = express();
// const port = 3001;

// app.use(cors());
// app.use(express.json());

// app.post('/rank-candidates', (req, res) => {
//     const { requiredRole, topN, skillsWeight } = req.body;
//     const mainDataFilePath = './skills_and_salaries_categories_p.csv';
//     const jobRolesDataFilePath = './job_skills.csv';
//     const testDataFilePath = './FinalResult.csv';

//     console.log(`Received request for role: ${requiredRole}, topN: ${topN}, skillsWeight: ${skillsWeight}`);

//     const pythonArgs = [
//         '-u',
//         'candidate_ranking.py',  // Python script that performs the ranking
//         requiredRole,
//         mainDataFilePath,
//         jobRolesDataFilePath,
//         testDataFilePath,
//         '--top_n', topN.toString(),
//         '--skills_weight', skillsWeight.toString()
//     ];

//     const pythonProcess = spawn('python', pythonArgs);

//     let result = '';
//     let error = '';

//     pythonProcess.stdout.on('data', (data) => {
//         result += data.toString();
//         console.log("Partial output:", data.toString());
//     });

//     pythonProcess.stderr.on('data', (data) => {
//         error += data.toString();
//         console.error(`Python script error: ${data}`);
//     });

//     pythonProcess.on('close', (code) => {
//         console.log(`Python script exited with code ${code}`);
//         if (code !== 0) {
//             console.error(`Error in Python script: ${error}`);
//             return res.status(500).json({ error: error });
//         }

//         try {
//             // Try to parse the result as JSON
//             const jsonStart = result.indexOf('[');
//             const jsonEnd = result.lastIndexOf(']');
//             if (jsonStart !== -1 && jsonEnd !== -1) {
//                 const jsonResult = JSON.parse(result.slice(jsonStart, jsonEnd + 1));
//                 res.json(jsonResult);
//             } else {
//                 console.error("Invalid JSON format");
//                 res.status(400).json({ error: "Invalid JSON format", details: result });
//             }
//         } catch (e) {
//             console.error(`Failed to parse result: ${e.message}`);
//             res.status(500).json({ error: 'Failed to parse result', details: result });
//         }
//     });
// });

// app.listen(port, () => {
//     console.log(`Server running on port ${port}`);
// });


const express = require('express');
const cors = require('cors');
const axios = require('axios');
const app = express();
const port = 3001;

app.use(cors());
app.use(express.json());

app.post('/rank-candidates', async (req, res) => {
    const { requiredRole, topN, skillsWeight, startDate, endDate } = req.body;
    const mainDataFilePath = './skills_and_salaries_categories_p.csv';
    const jobRolesDataFilePath = './job_skills.csv';
    const testDataFilePath = './Updated_Testing_Data.csv';

    console.log(`Received request for role: ${requiredRole}, topN: ${topN}, skillsWeight: ${skillsWeight}`);
    console.log(`Date filter: Start Date = ${startDate}, End Date = ${endDate}`);

    // Validate required fields
    if (!requiredRole || !topN || skillsWeight === undefined) {
        return res.status(400).json({
            error: "Missing required fields. Please provide 'requiredRole', 'topN', and 'skillsWeight'."
        });
    }

    // Validate optional date fields (if provided)
    if (startDate || endDate) {
        const startDateObj = startDate ? new Date(startDate) : null;
        const endDateObj = endDate ? new Date(endDate) : null;

        // Check if provided dates are valid
        if (startDate && isNaN(startDateObj.getTime())) {
            return res.status(400).json({
                error: "Invalid start date format. Please provide 'startDate' in YYYY-MM-DD format."
            });
        }

        if (endDate && isNaN(endDateObj.getTime())) {
            return res.status(400).json({
                error: "Invalid end date format. Please provide 'endDate' in YYYY-MM-DD format."
            });
        }

        // Ensure startDate is not after endDate
        if (startDateObj && endDateObj && startDateObj > endDateObj) {
            return res.status(400).json({
                error: "'startDate' cannot be after 'endDate'."
            });
        }
    }

    try {
        // Prepare the payload for the Python service
        const payload = {
            requiredRole,
            topN,
            skillsWeight,
            mainDataFilePath,
            jobRolesDataFilePath,
            testDataFilePath,
            startDate: startDate || null, // Send null if startDate is not provided
            endDate: endDate || null // Send null if endDate is not provided
        };

        // Make a POST request to the Python Flask service
        const response = await axios.post('https://backend-python-service.onrender.com/rank-candidates', payload, {
            timeout: 30000 // Optional: Set a timeout for the request
        });

        // Forward the response from the Python service to the frontend
        res.json(response.data);
    } catch (error) {
        console.error('Error:', error.message);

        // Handle different error scenarios
        if (error.response) {
            // The request was made and the server responded with a status code outside 2xx
            res.status(error.response.status).json({
                error: error.response.data.error || 'An error occurred in the Python service.'
            });
        } else if (error.request) {
            // The request was made but no response was received
            res.status(500).json({
                error: 'No response received from the Python service.'
            });
        } else {
            // Something happened in setting up the request
            res.status(500).json({
                error: 'An error occurred while setting up the request to the Python service.'
            });
        }
    }
});

app.listen(port, () => {
    console.log(`Server running on port ${port}`);
});
