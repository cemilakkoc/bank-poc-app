# Use an official Python runtime as a parent image
FROM python:3.12

# Set the working directory
WORKDIR /app

# Install required system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Hugging Face CLI and Transformers
RUN pip install --no-cache-dir flask transformers torch accelerate sentence-transformers

# Log in to Hugging Face using the provided token
ARG HUGGINGFACE_TOKEN
RUN huggingface-cli login --token ${HUGGINGFACE_TOKEN}

# Copy the Flask application code
COPY . .

# Expose port 5000 for the Flask app
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]
