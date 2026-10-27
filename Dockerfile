# Production Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for PDF processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create non-privileged system user for secure container execution
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/data/raw /app/data/processed /app/data/chroma_db /tmp/flashrank && \
    chown -R appuser:appuser /app /tmp/flashrank

# Copy source code and app
COPY src/ ./src/
COPY app/ ./app/
COPY data/create_sample_report.py ./data/create_sample_report.py
COPY data/download_sample_data.py ./data/download_sample_data.py
COPY data/eval_dataset.json ./data/eval_dataset.json
COPY README.md .

RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000
EXPOSE 8501

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
