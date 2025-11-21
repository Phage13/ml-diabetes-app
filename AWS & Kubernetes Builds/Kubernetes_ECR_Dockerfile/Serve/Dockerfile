# Use slim Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /opt/program

# Copy requirements first to leverage layer caching
COPY requirements.txt .

# Install build tools temporarily, install Python packages, then clean up
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential gcc python3-dev libffi-dev ca-certificates && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    # Explicitly pin critical libraries for reproducibility
    pip install --no-cache-dir flask==2.3.2 scikit-learn==1.5.1 pandas==2.1.1 joblib==1.3.2 numpy==1.26.0 && \
    # Show scikit-learn version during build for validation
    pip show scikit-learn && \
    apt-get purge -y --auto-remove build-essential gcc python3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy inference code
COPY inference.py inference.py

# Copy model artifact directly into /opt/program
COPY best_diabetes_model.pkl /opt/program/best_diabetes_model.pkl

# Environment settings
ENV PYTHONUNBUFFERED=TRUE

# Expose Flask port
EXPOSE 5000

# Run inference service
ENTRYPOINT ["python", "inference.py"]
