# Deployment Guide

This guide covers deployment options for the Biomedical Active Learning project.

## Quick Start with Docker Compose

### Prerequisites
- Docker Engine 20.10+ 
- Docker Compose 2.0+
- 4GB+ available RAM
- 10GB+ available disk space

### Launch All Services
```bash
# Clone/navigate to project directory
cd biomedical-active-learning

# Start all services (Streamlit app, Jupyter, Redis)
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Access Applications
- **Streamlit App**: http://localhost:8501
- **Jupyter Notebooks**: http://localhost:8888
- **Redis Cache**: localhost:6379

## Individual Service Deployment

### Streamlit Application Only
```bash
# Build and run Streamlit app
docker build -t biomedical-al .
docker run -p 8501:8501 \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/results:/app/results \
  biomedical-al
```

### Jupyter Notebooks Only
```bash
docker run -p 8888:8888 \
  -v $(pwd)/notebooks:/app/notebooks \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/results:/app/results \
  biomedical-al \
  jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

## Production Deployment

### Environment Variables
```bash
# Required
PYTHONPATH=/app/src
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Optional
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
JUPYTER_ENABLE_LAB=yes
```

### Resource Requirements

#### Minimum
- **CPU**: 2 cores
- **RAM**: 4GB
- **Storage**: 10GB
- **Network**: 1 Mbps

#### Recommended
- **CPU**: 4+ cores
- **RAM**: 8GB+
- **Storage**: 20GB+ SSD
- **Network**: 10+ Mbps

### Security Considerations

#### Container Security
- Runs as non-root user (`appuser`)
- Read-only file system where possible
- Minimal attack surface with slim base image
- Health checks for service monitoring

#### Network Security
```bash
# Custom network isolation
docker network create biomedical-network
docker run --network biomedical-network ...
```

#### Data Protection
```bash
# Mount data as read-only
-v $(pwd)/data:/app/data:ro

# Separate results volume
-v biomedical-results:/app/results
```

## Cloud Deployment

### AWS ECS
```yaml
# task-definition.json
{
  "family": "biomedical-al",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "4096",
  "containerDefinitions": [{
    "name": "streamlit-app",
    "image": "your-registry/biomedical-al:latest",
    "portMappings": [{"containerPort": 8501}],
    "healthCheck": {
      "command": ["CMD-SHELL", "curl -f http://localhost:8501/_stcore/health || exit 1"]
    }
  }]
}
```

### Google Cloud Run
```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT_ID/biomedical-al
gcloud run deploy biomedical-al \
  --image gcr.io/PROJECT_ID/biomedical-al \
  --platform managed \
  --memory 4Gi \
  --cpu 2 \
  --port 8501
```

### Azure Container Instances
```bash
az container create \
  --resource-group myResourceGroup \
  --name biomedical-al \
  --image your-registry/biomedical-al:latest \
  --cpu 2 \
  --memory 4 \
  --ports 8501
```

## Monitoring and Logging

### Health Checks
```bash
# Container health
curl http://localhost:8501/_stcore/health

# Service status
docker-compose ps
```

### Log Collection
```bash
# View application logs
docker-compose logs biomedical-al

# Stream logs
docker-compose logs -f --tail=100

# Export logs
docker-compose logs > app-logs.txt
```

### Performance Monitoring
```bash
# Resource usage
docker stats

# Service metrics
docker-compose top
```

## Troubleshooting

### Common Issues

#### Port Already in Use
```bash
# Check port usage
lsof -i :8501
netstat -tulpn | grep 8501

# Use different port
docker run -p 8502:8501 biomedical-al
```

#### Memory Issues
```bash
# Increase container memory
docker run --memory 8g biomedical-al

# Check memory usage
docker stats
```

#### Permission Errors
```bash
# Fix volume permissions
sudo chown -R $USER:$USER ./data ./results

# Check container user
docker exec -it container_name whoami
```

### Debug Mode
```bash
# Run with debug output
STREAMLIT_LOG_LEVEL=debug docker-compose up

# Interactive debugging
docker run -it --entrypoint bash biomedical-al
```

## Scaling

### Horizontal Scaling
```yaml
# docker-compose.yml
services:
  biomedical-al:
    deploy:
      replicas: 3
    # Load balancer configuration
```

### Load Balancing
```nginx
# nginx.conf
upstream biomedical_app {
    server localhost:8501;
    server localhost:8502;
    server localhost:8503;
}

server {
    listen 80;
    location / {
        proxy_pass http://biomedical_app;
    }
}
```

## Backup and Recovery

### Data Backup
```bash
# Backup results
docker run --rm -v biomedical_results:/data -v $(pwd):/backup alpine \
  tar czf /backup/results-backup.tar.gz -C /data .

# Backup models
cp -r ./models ./models-backup-$(date +%Y%m%d)
```

### Recovery
```bash
# Restore data
docker run --rm -v biomedical_results:/data -v $(pwd):/backup alpine \
  tar xzf /backup/results-backup.tar.gz -C /data
```

## Performance Optimization

### Image Size Optimization
- Multi-stage builds reduce final image size
- .dockerignore excludes unnecessary files
- Alpine base images for minimal footprint

### Runtime Optimization
```bash
# Disable file watching in production
STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

# Optimize memory usage
STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200

# Enable caching
STREAMLIT_SERVER_ENABLE_STATIC_SERVING=true
```

## CI/CD Integration

### GitHub Actions
```yaml
# .github/workflows/deploy.yml
name: Deploy
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build and push Docker image
        run: |
          docker build -t biomedical-al .
          docker push your-registry/biomedical-al:latest
```

### Automated Testing
```bash
# Test deployment
docker-compose -f docker-compose.test.yml up --abort-on-container-exit
```