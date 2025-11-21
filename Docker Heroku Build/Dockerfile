# Use a lightweight Python base image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies (important for pandas/numpy/scikit-learn builds)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code, model, and templates
COPY app.py .
COPY models/ ./models/
COPY templates/ ./templates/

# Expose Flask port (Heroku will map $PORT automatically)
EXPOSE 5000

# Default command (Heroku overrides with heroku.yml run:web)
CMD ["gunicorn", "app:app"]
