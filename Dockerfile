# Use official Python runtime
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set port for Cloud Run
ENV PORT=8080
ENV HOST=0.0.0.0

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose port
EXPOSE 8080

# Start the application
ENTRYPOINT ["python", "app.py"]
