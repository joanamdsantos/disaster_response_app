FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY ./app/ /app/

COPY ./models//classifier.pk /app/models/classifier.pk
COPY ./models//train_classifier.py /app/models/train_classifier.py 
COPY ./data/DisasterResponse.db /app/data/DisasterResponse.db
COPY ./data/process_data.py /app/data/process_data.py 
COPY ./data/disaster_categories.csv /app/data/disaster_categories.csv
COPY ./data/disaster_messages.csv /app/data/disaster_messages.csv

CMD ["python", "run.py"]
