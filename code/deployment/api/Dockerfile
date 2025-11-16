FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies required for OpenCV and other libraries
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY code/deployment/api/requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY code/deployment/api .

# Copy model files into the image so the API can load them at runtime
COPY models /app/models

RUN mkdir -p /app/models

EXPOSE 8000

CMD ["python", "-c", "import uvicorn; uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=True)"]
