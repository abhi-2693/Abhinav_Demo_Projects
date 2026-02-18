FROM python:3.11-slim

WORKDIR /app

# Install system-level dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (cache this layer!)
COPY requirements.txt /app/

# Install Python dependencies (only runs again if requirements.txt changes)
RUN pip install --no-cache-dir -r requirements.txt
    
# copy Project files and env file with credentials
COPY . /app
COPY .env /app/.env

# Set Python Module search path
ENV PYTHONPATH="/app:/app/src"

# Expose ports
EXPOSE 8501
EXPOSE 8000

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
