FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the application files
COPY southAsianHistoryResearchAgents_docker.py .
COPY search_tools_docker.py .

# Expose Streamlit port
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "southAsianHistoryResearchAgents_docker.py", "--server.address=0.0.0.0"]
