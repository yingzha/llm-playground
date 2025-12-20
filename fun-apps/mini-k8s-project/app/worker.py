# app/worker.py
import os
import time
import redis
import psycopg2

# --- CONFIGURATION ---
# We read these from Kubernetes Environment Variables
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PASS = os.getenv('DB_PASS', 'mysecretpassword')
DB_NAME = "job_db"

# --- CONNECT TO DATABASE ---
def get_db_connection():
    # Retry logic: The DB might not be ready immediately when the worker starts
    while True:
        try:
            conn = psycopg2.connect(
                host=DB_HOST, 
                database=DB_NAME, 
                user="postgres", 
                password=DB_PASS
            )
            return conn
        except psycopg2.OperationalError:
            print("⏳ Database not ready yet... waiting 2s")
            time.sleep(2)

print("🔌 Connecting to services...")
# Connect to Redis
r = redis.Redis(host=REDIS_HOST, port=6379, db=0)
# Connect to Postgres
conn = get_db_connection()

# --- INITIAL SETUP ---
# Create the table automatically so we don't have to do it manually
with conn.cursor() as cur:
    cur.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id SERIAL PRIMARY KEY,
            image_url TEXT,
            status TEXT,
            result TEXT
        );
    """)
    conn.commit()
print("✅ Table 'jobs' is ready.")

# --- THE WORKER LOOP ---
def run_worker():
    print("👷 Worker started. Waiting for jobs in 'image_queue'...")
    while True:
        # 1. BLOCK here until Redis gives us a job ID (0 means wait forever)
        # brpop returns a tuple: (queue_name, data)
        queue, job_id_bytes = r.brpop("image_queue", timeout=0)
        job_id = int(job_id_bytes)

        print(f"📥 Received Job ID: {job_id}")

        # 2. Update DB: Set to PROCESSING
        with conn.cursor() as cur:
            cur.execute("UPDATE jobs SET status = 'PROCESSING' WHERE id = %s", (job_id,))
            conn.commit()

        # 3. Simulate heavy work (AI processing)
        time.sleep(5) 
        fake_result_url = f"processed_image_{job_id}.jpg"

        # 4. Update DB: Set to COMPLETED
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE jobs SET status = 'COMPLETED', result = %s WHERE id = %s", 
                (fake_result_url, job_id)
            )
            conn.commit()
        
        print(f"✨ Job {job_id} Finished!")

if __name__ == "__main__":
    run_worker()