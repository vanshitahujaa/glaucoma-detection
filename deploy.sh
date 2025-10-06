#!/bin/bash

# Retinal Disease Classification Deployment Script
echo "Starting Retinal Disease Classification Application..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python3 is not installed. Please install it first."
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "Node.js is not installed. Please install it first."
    exit 1
fi

# Function to start the backend
start_backend() {
    echo "Starting backend server..."
    cd backend
    python3 main.py &
    BACKEND_PID=$!
    echo "Backend started with PID: $BACKEND_PID"
    cd ..
}

# Function to start the frontend
start_frontend() {
    echo "Starting frontend server..."
    cd frontend
    npm start &
    FRONTEND_PID=$!
    echo "Frontend started with PID: $FRONTEND_PID"
    cd ..
}

# Start the application
start_backend
sleep 5  # Wait for backend to initialize
start_frontend

echo "Application started successfully!"
echo "- Backend: http://localhost:8000"
echo "- Frontend: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop the application"

# Wait for user to stop the application
wait