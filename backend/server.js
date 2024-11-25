const express = require('express');
const cors = require('cors');
const axios = require('axios');
const app = express();
const port = 3001;

app.use(cors());
app.use(express.json());

app.post('/rank-candidates', async (req, res) => {
    const { requiredRole, topN, startDate, endDate, maxExperience, minExperience, mandatorySkills, optionalSkills } = req.body;
    const mainDataFilePath = './skills_and_salaries_categories_p.csv';
    const jobRolesDataFilePath = './job_skills.csv';
    const testDataFilePath = './Results.csv';

    console.log(`Received request for role: ${requiredRole}, topN: ${topN}`);
    console.log(`Date filter: Start Date = ${startDate}, End Date = ${endDate}`);
    console.log(`Max Experience: ${maxExperience}`);
    console.log(`Mandatory Skills: ${mandatorySkills}`);
    console.log(`Optional Skills: ${optionalSkills}`);

    // Validate required fields
    if (!requiredRole || !topN) {
        return res.status(400).json({
            error: "Missing required fields. Please provide 'requiredRole' and 'topN'."
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
            mainDataFilePath,
            jobRolesDataFilePath,
            testDataFilePath,
            startDate: startDate || null,
            endDate: endDate || null,
            maxExperience: maxExperience || null,
            minExperience: minExperience || null,
            mandatorySkills: mandatorySkills || [],
            optionalSkills: optionalSkills || []
        };

        // Make a POST request to the local Python Flask service
        const response = await axios.post('https://backend-python-service.onrender.com/rank-candidates', payload, {
            timeout: 90000 // Optional: Set a timeout for the request
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