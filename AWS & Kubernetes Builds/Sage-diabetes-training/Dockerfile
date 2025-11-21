FROM python:3.9-slim

WORKDIR /opt/program

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy training and inference scripts
COPY train.py inference.py ./

# SageMaker expects the training toolkit to manage lifecycle
ENV PYTHONUNBUFFERED=TRUE

# Use SageMaker training toolkit entrypoint
ENTRYPOINT ["python", "train.py"]