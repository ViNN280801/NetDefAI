FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
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

CMD ["python", "src/alert_service/alert_service.py"]
