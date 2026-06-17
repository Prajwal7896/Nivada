FROM python:3.10

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip

RUN pip install fastapi uvicorn sqlalchemy psycopg2-binary \
    torch transformers sentence-transformers rank-bm25 \
    scikit-learn pandas numpy mlflow werkzeug

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
