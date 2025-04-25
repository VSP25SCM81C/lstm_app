FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV HOST=0.0.0.0

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure image path exists
RUN mkdir -p static/images

EXPOSE 8080

# Use Gunicorn to serve the app
ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]


