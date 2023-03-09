FROM node:14-alpine

WORKDIR /app
# Copy the project files
COPY . .

# Install dependencies
RUN npm install

# Build the project
RUN npm run build

# Expose the port
EXPOSE 3000

# Start the application
CMD ["npm", "start"]

EXPOSE 5000