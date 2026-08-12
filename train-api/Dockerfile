# train-api/Dockerfile — statsmodels/scikit stack, no GPU needed (small monthly time series)
FROM python:3.11-slim

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r /app/requirements.txt

COPY . /app

EXPOSE 5010

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:5010/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5010"]
