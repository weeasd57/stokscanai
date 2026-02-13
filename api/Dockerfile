FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code into an 'api' folder to preserve package structure
COPY . ./api/

# Hugging Face Spaces expects the app to listen on port 7860 by default.
# Keep PORT override support for other platforms.
ENV PORT=7860
EXPOSE 7860

# Start command pointing to the api package
CMD ["sh", "-c", "python -m uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
