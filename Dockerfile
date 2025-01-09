# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set up working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    curl \
    software-properties-common && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the application files
COPY southAsianHistoryResearch_Render.py .

# If you have a search_tools.py file, copy it as well
# COPY search_tools.py .

# Create and configure streamlit directory
RUN mkdir -p /root/.streamlit
COPY .streamlit/config.toml /root/.streamlit/config.toml

# Expose Streamlit port
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "southAsianHistoryResearch_Render.py", "--server.port=8501", "--server.address=0.0.0.0"]