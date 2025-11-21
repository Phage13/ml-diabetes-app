FROM python:3.9-slim

WORKDIR /opt/program

# Copy requirements first to leverage layer caching
COPY requirements.txt requirements.txt

# Install build tools temporarily, install Python packages (including Flask), then clean up
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential gcc python3-dev libffi-dev ca-certificates && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir flask && \
    apt-get purge -y --auto-remove build-essential gcc python3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy training code and dataset
COPY train.py train.py
COPY diabetes_prediction_dataset.csv diabetes_prediction_dataset.csv
COPY inference.py inference.py

ENV PYTHONUNBUFFERED=TRUE

# Default command (overridden by Kubernetes Job manifest)
ENTRYPOINT ["python", "train.py"]