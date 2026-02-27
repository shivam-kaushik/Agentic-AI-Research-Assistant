# Use an official lightweight Python image.
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the local codebase to the container
COPY . .

# Expose the standard Streamlit port
EXPOSE 8501

# Provide an environment variable to specify GCP credentials discovery
# (Cloud Run handles this automatically, but included for local testing)
ENV GOOGLE_CLOUD_PROJECT="benchspark-1771447466"

# Run the Streamlit application
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
