################################
# ISB AMPBA - Term 2
# CT1 - MLOPS Assignment - Deploy Dockerized Streamlit App Using AWS EC2, RDS, ECR, IAM
# Name - Abhinav Paul
# PGID - 1242055
################################

# 1. Use a stable base image with Python 3.11 (not 3.13, to avoid package build issues)
FROM python:3.11-slim

# 2. Install system packages required for building Python libraries like pandas, plotly, etc.
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libpq-dev \
    python3-dev && \
    rm -rf /var/lib/apt/lists/*


# 3. Set the working directory inside the container to /app
WORKDIR /app

# 4. Copy all project files from the current host directory into the container's /app
COPY . /app

# 5. Upgrade pip, setuptools, and wheel to ensure compatibility with modern Python packaging (PEP 517/518)
RUN pip install --upgrade pip setuptools wheel

# 6. Install Python dependencies listed in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 7. Expose port 8501 so Docker knows your app runs on this port (Streamlit default)
EXPOSE 8501

# 8. Set the default command to run your Streamlit app when the container starts
CMD ["streamlit", "run", "api.py", "--server.port=", "--server.address=0.0.0.0"]
