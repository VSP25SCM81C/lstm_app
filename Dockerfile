# Use official Python runtime
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


# Set environment variables (optional)
ENV PORT=8080
ENV HOST=0.0.0.0


WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY . .

# Cloud Run expects the container to listen on $PORT (already handled in your app.py)
CMD ["python", "app.py"]
