# FROM node:14 as build
# WORKDIR /app
# COPY package*.json ./
# RUN npm install
# COPY . .
# RUN npm run build

# FROM nginx:alpine
# COPY --from=build /app/build /usr/share/nginx/html
# EXPOSE 3000
# CMD ["nginx", "-g", "daemon off;"]

# Use a node image for the build process
FROM node:14 as build

WORKDIR /app

# Copy package.json and package-lock.json
COPY ./frontend/package*.json ./ 

# Install dependencies
RUN npm install

# Copy the rest of the application code
COPY ./ . 

# Build the application
RUN npm run build

# Use nginx to serve the app
FROM nginx:alpine

# Copy the built files to the Nginx directory
COPY --from=build /app/build /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
