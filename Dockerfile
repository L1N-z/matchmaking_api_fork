# Use a lightweight Python base image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory
WORKDIR /app

# Copy only the requirements first to leverage Docker layer caching
# This makes future deployments much faster if you haven't changed dependencies
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Fly.io defaults to port 8080. 
# We expose it here and tell Uvicorn to listen on it.
EXPOSE 8080

# Command to run the application
# Replace 'main' with your actual filename if it isn't main.py
CMD ["uvicorn", "matchmaking_api:app", "--host", "0.0.0.0", "--port", "8080"]
