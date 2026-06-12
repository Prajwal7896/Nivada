# Base image
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy project
COPY . /app

# Install dependencies
RUN pip install --upgrade pip

RUN pip install fastapi uvicorn sqlalchemy psycopg2-binary \
    torch transformers sentence-transformers rank-bm25 \
    scikit-learn pandas numpy mlflow werkzeug

# Expose port
EXPOSE 8000

# Start app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]