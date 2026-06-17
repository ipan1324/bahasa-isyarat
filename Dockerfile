FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Make startup script executable
RUN chmod +x start.sh

# Expose port (Railway overrides this with $PORT anyway)
EXPOSE 8080

# Start the app via shell script so $PORT is always expanded
CMD ["sh", "start.sh"]
