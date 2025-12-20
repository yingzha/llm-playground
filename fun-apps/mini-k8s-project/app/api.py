# app/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import redis
import psycopg2
import os
import time

app = FastAPI()

# --- CONFIGURATION ---
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PASS = os.getenv('DB_PASS', 'mysecretpassword')

# --- CONNECT TO SERVICES ---
r = redis.Redis(host=REDIS_HOST, port=6379, db=0)

def get_db_connection():
    try:
        return psycopg2.connect(
            host=DB_HOST, database="job_db", user="postgres", password=DB_PASS
        )
    except Exception as e:
        print(f"DB Connection failed: {e}")
        return None

# --- DATA MODEL ---
class JobRequest(BaseModel):
    image_url: str

# --- API ENDPOINT ---
@app.post("/submit")
def submit_job(job: JobRequest):
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database offline")

    try:
        # 1. Save to DB (Source of Truth)
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO jobs (image_url, status) VALUES (%s, 'PENDING') RETURNING id",
                (job.image_url,)
            )
            job_id = cur.fetchone()[0]
            conn.commit()

        # 2. Push to Redis (The Signal)
        r.lpush("image_queue", job_id)

        return {"job_id": job_id, "status": "queued", "message": "Job submitted successfully"}

    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

# Add a health check
@app.get("/")
def home():
    return {"status": "API is running"}