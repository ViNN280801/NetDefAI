FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y netcat-openbsd --no-install-recommends \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY patterns/ ./patterns/
COPY config.yaml .

RUN mkdir -p logs

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "src.api_gateway.app:app", "--host", "0.0.0.0", "--port", "8000"]
