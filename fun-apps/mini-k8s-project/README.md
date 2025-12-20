# Mini K8s Image Processing Queue

A microservices demo showing Kubernetes orchestration with Redis job queuing.

## Architecture

API → Redis Queue → Worker → PostgreSQL

- **API**: FastAPI server that queues jobs
- **Worker**: Background processor
- **Redis**: Job queue
- **PostgreSQL**: Job storage

## Quick Start

```bash
# 1. Build image
cd app && docker build -t my-image-worker:v1 .

# 2. Deploy everything
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/postgres.yaml
kubectl apply -f k8s/redis.yaml
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/worker-deployment.yaml

# 3. Get API URL
minikube service image-api-service --url

# 4. Submit a job
curl -X POST "<API_URL>/submit" \
     -H "Content-Type: application/json" \
     -d '{"image_url": "http://example.com/cat.jpg"}'

# 5. Watch worker process it
kubectl logs -f -l app=image-worker
```

## Useful Commands

```bash
# View all pods
kubectl get pods

# Scale workers
kubectl scale deployment image-worker --replicas=3

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
│   ├── worker.py       # Job processor
│   └── Dockerfile      # Container image
└── k8s/
    ├── secrets.yaml    # Secrets & config
    ├── postgres.yaml   # Database
    ├── redis.yaml      # Queue
    ├── api-deployment.yaml    # API deployment
    └── worker-deployment.yaml # Worker deployment
```

## Troubleshooting

**Worker not processing jobs?**
- Check: `kubectl logs -l app=image-worker`
- Verify command is set in [worker-deployment.yaml](k8s/worker-deployment.yaml#L19-L20)

**Can't connect to API?**
- Get URL: `minikube service image-api-service --url`
- Check logs: `kubectl logs -l app=image-api`

**Database errors?**
- Check pod: `kubectl get pods -l app=postgres-db`
- Verify secret: `kubectl get secret app-secrets -o yaml`
