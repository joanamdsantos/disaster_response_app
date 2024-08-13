FROM python:3.6-alpine3.13
RUN apk add --no-cache \
    build-base \
    gcc \
    musl-dev \
    python3-dev \
    libffi-dev \
    openssl-dev
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "run.py"]