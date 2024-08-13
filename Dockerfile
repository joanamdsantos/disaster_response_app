FROM python:3.13-alpine
WORKDIR /app
3. ADD . /app
4. RUN pip install -r requirements.txt
5. CMD ["python","run.py"]