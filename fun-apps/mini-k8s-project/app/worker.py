# app/worker.py
import os
import time
import redis
import psycopg2
import socket
from datetime import datetime, timedelta

# --- CONFIGURATION ---
# We read these from Kubernetes Environment Variables
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PASS = os.getenv('DB_PASS', 'mysecretpassword')
DB_NAME = "job_db"
WORKER_NAME = socket.gethostname()  # Gets the pod name in Kubernetes
JOB_RECOVERY_INTERVAL = 30  # Check for orphaned jobs every 30 seconds

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

# --- CONNECT TO REDIS ---
def get_redis_connection():
    """Connect to Redis with retry logic"""
    while True:
        try:
            r = redis.Redis(host=REDIS_HOST, port=6379, db=0, socket_connect_timeout=5)
            r.ping()  # Test connection
            return r
        except (redis.ConnectionError, redis.TimeoutError) as e:
            print(f"⏳ Redis not ready yet... waiting 2s ({e})")
            time.sleep(2)

# --- JOB RECOVERY ---
def recover_orphaned_jobs(conn, r):
    """Find PENDING jobs in DB and requeue them to Redis"""
    try:
        with conn.cursor() as cur:
            # Find jobs that are PENDING (should be in queue but might not be)
            cur.execute("SELECT id FROM jobs WHERE status = 'PENDING'")
            orphaned_jobs = cur.fetchall()

            if orphaned_jobs:
                print(f"🔄 Found {len(orphaned_jobs)} orphaned PENDING jobs, requeuing...")
                for (job_id,) in orphaned_jobs:
                    r.lpush("image_queue", job_id)
                    print(f"   ↪ Requeued job {job_id}")
    except Exception as e:
        print(f"⚠️  Job recovery failed: {e}")

print("🔌 Connecting to services...")
# Connect to Redis
r = get_redis_connection()
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
    global conn, r
    print(f"👷 [{WORKER_NAME}] Worker started. Waiting for jobs in 'image_queue'...")

    last_recovery_check = time.time()

    while True:
        try:
            # Periodic job recovery check
            if time.time() - last_recovery_check > JOB_RECOVERY_INTERVAL:
                recover_orphaned_jobs(conn, r)
                last_recovery_check = time.time()

            # 1. BLOCK here until Redis gives us a job ID (timeout allows periodic recovery checks)
            # brpop returns a tuple: (queue_name, data) or None if timeout
            result = r.brpop("image_queue", timeout=5)

            if result is None:
                # Timeout, no job available - loop to check for orphaned jobs
                continue

            _, job_id_bytes = result  # _ is the queue name, we don't need it
            job_id = int(job_id_bytes)

            print(f"📥 [{WORKER_NAME}] Received Job ID: {job_id}")

            # 2. Update DB: Set to PROCESSING
            try:
                with conn.cursor() as cur:
                    cur.execute("UPDATE jobs SET status = 'PROCESSING' WHERE id = %s", (job_id,))
                    conn.commit()
            except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
                print(f"⚠️  DB connection lost, reconnecting... ({e})")
                conn = get_db_connection()
                with conn.cursor() as cur:
                    cur.execute("UPDATE jobs SET status = 'PROCESSING' WHERE id = %s", (job_id,))
                    conn.commit()

            # 3. Simulate heavy work (AI processing)
            time.sleep(5)
            fake_result_url = f"processed_image_{job_id}.jpg"

            # 4. Update DB: Set to COMPLETED
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE jobs SET status = 'COMPLETED', result = %s WHERE id = %s",
                        (fake_result_url, job_id)
                    )
                    conn.commit()
            except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
                print(f"⚠️  DB connection lost, reconnecting... ({e})")
                conn = get_db_connection()
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE jobs SET status = 'COMPLETED', result = %s WHERE id = %s",
                        (fake_result_url, job_id)
                    )
                    conn.commit()

            print(f"✨ [{WORKER_NAME}] Job {job_id} Finished!")

        except (redis.ConnectionError, redis.TimeoutError) as e:
            print(f"⚠️  Redis connection lost: {e}")
            print(f"🔄 Reconnecting to Redis in 5 seconds...")
            time.sleep(5)
            r = get_redis_connection()
            print(f"✅ Reconnected to Redis!")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            time.sleep(2)  # Brief pause before retrying

if __name__ == "__main__":
    run_worker()