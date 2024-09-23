# Build stage
FROM node:14-alpine AS build

# Install build dependencies
RUN apk add --no-cache python3 py3-pip build-base python3-dev openblas-dev

# Set the working directory
WORKDIR /usr/src/app

# Copy package.json and package-lock.json
COPY package*.json ./

# Install Node.js dependencies
RUN npm install --production

# Copy Python requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Final stage
FROM node:14-alpine

# Install runtime dependencies
RUN apk add --no-cache python3 py3-pip openblas

# Set the working directory
WORKDIR /usr/src/app

# Copy Node.js dependencies from the build stage
COPY --from=build /usr/src/app/node_modules ./node_modules

# Copy Python packages from the build stage
COPY --from=build /usr/local/lib/python3.* /usr/local/lib/
COPY --from=build /usr/lib/python3.* /usr/lib/

# Copy the rest of the application code
COPY . .

# Make port 3001 available outside the container
EXPOSE 3001

# Define the command to run the app
CMD ["node", "server.js"]