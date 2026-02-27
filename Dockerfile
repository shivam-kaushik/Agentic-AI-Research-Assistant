# Use an official lightweight Python image.
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for matplotlib and other packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the local codebase to the container
COPY . .

# Create outputs directory for generated files
RUN mkdir -p outputs local_storage/sessions local_storage/hitl_checkpoints

# Expose port (Cloud Run will set PORT env var, default to 8080)
EXPOSE 8080

# Set environment variables for GCP and Streamlit
ENV GOOGLE_CLOUD_PROJECT="queryquest-1771952465" \
    GOOGLE_CLOUD_QUOTA_PROJECT="queryquest-1771952465" \
    PORT=8080 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true

# Run the Streamlit application (use PORT env var for Cloud Run compatibility)
CMD streamlit run app/main_v2.py \
    --server.port=$PORT \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.serverAddress=0.0.0.0 \
    --browser.gatherUsageStats=false \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false
