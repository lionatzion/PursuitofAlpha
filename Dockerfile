FROM python:3.10-slim AS base
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "pipelines/full_pipeline.py"]

# Test target includes pytest for running the suite inside Docker
FROM base AS test
RUN pip install --no-cache-dir pytest
