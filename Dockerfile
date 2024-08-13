FROM python:3.6-alpine3.13
WORKDIR /app
ADD . /app
RUN pip install -r requirements.txt
CMD ["python","run.py"]