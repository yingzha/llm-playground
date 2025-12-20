# Testing Worker Resilience

## Features Implemented

✅ **Redis Reconnection** - Workers automatically reconnect if Redis fails
✅ **Job Recovery** - Orphaned PENDING jobs are requeued every 30 seconds
✅ **DB Reconnection** - Handles PostgreSQL connection failures
✅ **Graceful Error Handling** - Workers continue running on errors

## Test Scenarios

### Test 1: Redis Restart (Simulated Failure)

```bash
# Watch worker logs in one terminal
kubectl logs -l app=image-worker -f

# In another terminal, restart Redis
kubectl delete pod -l app=redis-cache

# Observe: Workers detect Redis failure, reconnect automatically
# Expected output:
# ⚠️  Redis connection lost: ...
# 🔄 Reconnecting to Redis in 5 seconds...
# ✅ Reconnected to Redis!
```

### Test 2: Job Recovery

```bash
# 1. Submit a job via API
curl -X POST "<API_URL>/submit" \
     -H "Content-Type: application/json" \
     -d '{"image_url": "http://example.com/test.jpg"}'

# 2. Immediately delete Redis (before worker picks up job)
kubectl delete pod -l app=redis-cache

# 3. Watch worker logs - within 30 seconds you'll see:
# 🔄 Found 1 orphaned PENDING jobs, requeuing...
#    ↪ Requeued job 15

# 4. Job will be processed once Redis is back
```

### Test 3: Database Restart

```bash
# Restart PostgreSQL while jobs are processing
kubectl delete pod -l app=postgres-db

# Workers will reconnect and continue processing
# Expected output:
# ⚠️  DB connection lost, reconnecting...
```

### Test 4: Scale Under Load

```bash
# Submit multiple jobs
for i in {1..10}; do
  curl -X POST "<API_URL>/submit" \
       -H "Content-Type: application/json" \
       -d "{\"image_url\": \"http://example.com/img$i.jpg\"}"
done

# Scale workers up/down during processing
kubectl scale deployment image-worker --replicas=5
kubectl scale deployment image-worker --replicas=2

# Jobs continue processing without loss
```

## Monitoring

```bash
# Watch all worker logs with timestamps
kubectl logs -l app=image-worker -f --timestamps

# Check job status in database
kubectl exec -it <postgres-pod> -- psql -U postgres -d job_db
SELECT id, status, image_url FROM jobs ORDER BY id DESC LIMIT 10;

# Check Redis queue length
kubectl exec -it <redis-pod> -- redis-cli LLEN image_queue
```

## What's NOT Handled

- Jobs that fail during processing (no retry logic yet)
- Long-running job timeout (processing can hang indefinitely)
- Concurrent job recovery (multiple workers might requeue same job)

## Future Improvements

- Add job retry with exponential backoff
- Implement job timeout/cancellation
- Add distributed lock for job recovery
- Use Redis Streams instead of Lists for better reliability
- Add monitoring and alerting (Prometheus/Grafana)
