# Mini K8s Image Processing Queue

Production-ready microservices with Kubernetes orchestration, Redis job queuing, HTTPS Ingress, and fault tolerance.

## Architecture

```
HTTPS Request → Ingress → API → Redis Queue → Workers (3x) → PostgreSQL
```

**Components:**
- **Ingress**: HTTPS/TLS termination with nginx
- **API**: FastAPI server (ClusterIP)
- **Workers**: 3 replicas with auto-recovery
- **Redis**: Job queue with reconnection logic
- **PostgreSQL**: Persistent job storage

## Quick Start

```bash
# 1. Build image
cd app && docker build -t my-image-worker:v3 .
minikube image load my-image-worker:v3

# 2. Enable Ingress
minikube addons enable ingress

# 3. Deploy everything
kubectl apply -f k8s/resource-quota.yaml  # Set resource limits
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/postgres.yaml
kubectl apply -f k8s/redis.yaml
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/worker-deployment.yaml
kubectl apply -f k8s/ingress.yaml

# 4. Setup local domain
echo "127.0.0.1 image-api.local" | sudo tee -a /etc/hosts

# 5. Start tunnel (in separate terminal)
minikube tunnel

# 6. Submit a job via HTTPS
curl -k -X POST "https://image-api.local/submit" \
     -H "Content-Type: application/json" \
     -d '{"image_url": "http://example.com/cat.jpg"}'

# 7. Watch workers process it
kubectl logs -f -l app=image-worker
```

**Output:** You'll see which worker processes which job:
```
📥 [image-worker-79fd8bd479-25cww] Received Job ID: 5
✨ [image-worker-79fd8bd479-25cww] Job 5 Finished!
```

## Useful Commands

```bash
# View all pods
kubectl get pods

# Scale workers (limited by ResourceQuota)
kubectl scale deployment image-worker --replicas=5

# Check resource usage
kubectl describe quota mini-k8s-quota
kubectl top pods

# View logs
kubectl logs -l app=image-worker -f
kubectl logs -l app=image-api -f

# Cleanup
kubectl delete -f k8s/
```

## Configuration

Credentials are in [k8s/secrets.yaml](k8s/secrets.yaml):
- **Secret**: Database password (base64 encoded)
- **ConfigMap**: Service hostnames and settings

**Important**: Add `k8s/secrets.yaml` to `.gitignore`!

## Project Structure

```
├── app/
│   ├── api.py          # FastAPI server
│   ├── worker.py       # Job processor with resilience
│   └── Dockerfile      # Container image
└── k8s/
    ├── resource-quota.yaml    # Resource limits (prevents runaway scaling)
    ├── secrets.yaml           # Secrets & ConfigMap
    ├── ingress.yaml           # HTTPS Ingress
    ├── postgres.yaml          # Database
    ├── redis.yaml             # Queue
    ├── api-deployment.yaml    # API deployment (ClusterIP)
    └── worker-deployment.yaml # Worker deployment (3 replicas)
```

## Features

✅ **HTTPS with Ingress** - Production-standard TLS termination
✅ **Horizontal Scaling** - 3 worker replicas for parallel processing
✅ **Resource Quotas** - Prevents accidental runaway scaling (max 15 pods, 2 CPU cores)
✅ **Fault Tolerance** - Workers auto-reconnect to Redis/PostgreSQL
✅ **Job Recovery** - Orphaned jobs automatically requeued every 30s
✅ **Secrets Management** - Kubernetes Secrets for credentials
✅ **Worker Identification** - Logs show which worker processes each job

## Troubleshooting

**Can't reach API via HTTPS?**
- Ensure `minikube tunnel` is running in another terminal
- Check domain: `cat /etc/hosts` should have `127.0.0.1 image-api.local`
- Verify ingress: `kubectl get ingress` (should show ADDRESS)

**Certificate error in browser?**
- Expected with self-signed certs - click "Advanced" → "Proceed"
- Use `-k` flag with curl to skip verification

**Worker not processing jobs?**
- Check logs: `kubectl logs -l app=image-worker`
- Workers auto-recover from Redis failures every 30s

**Redis/DB connection issues?**
- Workers will auto-reconnect - check logs for "Reconnected" messages
- Kubernetes automatically restarts failed pods

**Can't scale beyond a certain number of workers?**
- Check quota: `kubectl describe quota mini-k8s-quota`
- ResourceQuota limits: 15 pods max, 2 CPU cores total
- Events will show: "exceeded quota" if you hit limits
