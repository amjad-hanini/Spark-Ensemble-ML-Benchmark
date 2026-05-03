# Use a lightweight Python base image
FROM python:3.12-slim

# Install Java (Required for Apache Spark to run) & wget for data downloads
RUN apt-get update && \
    apt-get install -y default-jre wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files into the container
COPY . .

# Command to execute the benchmark pipeline when the container starts
CMD ["python", "benchmark.py"]
