FROM python:3.9-slim

WORKDIR /app

# Added fastapi and uvicorn for the API
RUN pip install redis psycopg2-binary fastapi uvicorn requests

# Copy ALL code (worker.py AND api.py)
COPY . .

# We don't specify CMD here anymore, because we will override it 
# in the Kubernetes YAML (one for api, one for worker)