FROM python:3.8-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

RUN python -m nltk.downloader punkt
RUN pip install xgboost==1.3.3
RUN pip install scikit-learn==0.24.2

COPY ./app/run.py /app/run.py
COPY ./app/templates/go.html /app/templates/go.html
COPY ./app/templates/master.html /app/templates/master.html

COPY ./models/classifier.pk /app/models/classifier.pk
COPY ./models/train_classifier.py /app/models/train_classifier.py 
COPY ./data/DisasterResponse.db /app/data/DisasterResponse.db
COPY ./data/process_data.py /app/data/process_data.py 
COPY ./data/disaster_categories.csv /app/data/disaster_categories.csv
COPY ./data/disaster_messages.csv /app/data/disaster_messages.csv

CMD ["python", "run.py"]
