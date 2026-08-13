# Railway / production image for Aegis.
# Keeps the same runtime as local: Python 3.12 + web/server.py on $PORT.
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Shapely / scientific wheels usually ship prebuilt; keep a thin toolchain
# available in case a dependency needs a source build on Linux.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY . .

# Railway injects $PORT at runtime. Default to 8080 only for local docker runs.
ENV PORT=8080
EXPOSE 8080

# Bound to 0.0.0.0:$PORT inside web/server.py
CMD ["python", "web/server.py"]
