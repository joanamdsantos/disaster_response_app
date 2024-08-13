FROM python:3.6-alpine3.13
WORKDIR /app
3. ADD . /app
4. RUN pip install -r requirements.txt
5. CMD ["python","run.py"]