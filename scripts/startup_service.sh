#!/bin/bash

# Set strict error handling
set -euo pipefail

echo "Starting DeepVisionXplain API..."

# Change to the application directory
cd /app

# Check if we're running as the correct user
echo "Running as user: $(whoami)"

# Start the FastAPI application
echo "Starting FastAPI server on port 8000..."
exec python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --no-access-log --log-level error