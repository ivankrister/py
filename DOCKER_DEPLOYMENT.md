# Docker Deployment Guide

## Overview

This guide explains how to deploy the Receipt OCR API using Docker and Docker Compose for production environments.

## Prerequisites

- Docker Engine 20.10+ 
- Docker Compose 2.0+
- At least 2GB RAM
- 1GB free disk space

## Quick Start

### 1. Basic Deployment

```bash
# Clone or navigate to the project directory
cd /path/to/python_orc

# Build and start the application
docker-compose up -d

# Check if it's running
curl http://localhost:8000/health
```

### 2. Production Deployment

```bash
# Use the deployment script
./deploy.sh
```

## Configuration

### Environment Variables

Copy `env.example` to `.env` and modify as needed:

```bash
cp env.example .env
# Edit .env with your production settings
```

### Key Configuration Options

- `HOST`: Bind address (default: 0.0.0.0)
- `PORT`: Port number (default: 8000)
- `DEBUG`: Enable debug mode (default: false)
- `CORS_ORIGINS`: Allowed origins for CORS

## Docker Compose Profiles

### Basic Profile (Default)
```bash
docker-compose up -d
```
Starts only the OCR API service.

### Production Profile
```bash
docker-compose --profile production up -d
```
Includes Nginx reverse proxy with SSL support.

## Production Considerations

### 1. Reverse Proxy (Nginx)

The included Nginx configuration provides:
- Load balancing
- Rate limiting (10 requests/second)
- Security headers
- File upload size limits
- SSL termination (when configured)

### 2. SSL/TLS Setup

1. Place your SSL certificates in the `ssl/` directory:
   ```
   ssl/
   ├── cert.pem
   └── key.pem
   ```

2. Uncomment the HTTPS server block in `nginx.conf`

3. Update the server_name to your domain

### 3. Domain Configuration

Update the following files with your domain:
- `docker-compose.yml`: Traefik labels
- `nginx.conf`: server_name directive
- `.env`: CORS_ORIGINS and ALLOWED_HOSTS

### 4. Monitoring and Logging

#### Health Checks
- Container health: `docker-compose ps`
- Application health: `curl http://localhost:8000/health`

#### Logs
```bash
# View application logs
docker-compose logs -f ocr-api

# View Nginx logs (if using production profile)
docker-compose logs -f nginx

# View all logs
docker-compose logs -f
```

#### Log Persistence
Application logs are stored in `./logs/` directory (mounted as volume).

### 5. Performance Tuning

#### Memory Optimization
For production, adjust Docker memory limits:

```yaml
services:
  ocr-api:
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 512M
```

#### CPU Optimization
OCR processing is CPU-intensive. Consider:
- Using multiple replicas
- CPU limits and reservations
- Horizontal scaling with load balancer

### 6. Security

#### Network Security
- Use custom Docker networks
- Restrict exposed ports
- Implement firewall rules

#### Application Security
- Configure CORS properly
- Set up rate limiting
- Use HTTPS in production
- Regular security updates

## Scaling

### Horizontal Scaling
```bash
# Scale to 3 replicas
docker-compose up -d --scale ocr-api=3
```

### Load Balancing
Use the included Nginx configuration or external load balancers like:
- AWS Application Load Balancer
- Google Cloud Load Balancer
- Azure Load Balancer

## Deployment Strategies

### Blue-Green Deployment
1. Deploy new version with different service name
2. Test the new version
3. Switch traffic to new version
4. Remove old version

### Rolling Updates
```bash
# Update with zero downtime
docker-compose up -d --force-recreate --no-deps ocr-api
```

## Troubleshooting

### Common Issues

1. **Container won't start**
   ```bash
   docker-compose logs ocr-api
   ```

2. **OCR not working**
   - Check Tesseract installation in container
   - Verify image preprocessing

3. **High memory usage**
   - Monitor with `docker stats`
   - Implement image processing optimization

4. **Slow response times**
   - Check CPU usage
   - Optimize image preprocessing
   - Implement caching

### Debug Commands

```bash
# Enter running container
docker-compose exec ocr-api bash

# Check Tesseract installation
docker-compose exec ocr-api tesseract --version

# Check Python packages
docker-compose exec ocr-api pip list

# Monitor resources
docker stats
```

## Backup and Recovery

### Data Backup
Currently, the application is stateless. Back up:
- Configuration files (.env, docker-compose.yml)
- SSL certificates
- Custom configuration

### Disaster Recovery
1. Store Docker images in a registry
2. Backup configuration to version control
3. Document deployment procedures
4. Test recovery procedures regularly

## Maintenance

### Updates
```bash
# Update application
git pull
docker-compose build --no-cache
docker-compose up -d

# Update base images
docker-compose pull
docker-compose up -d
```

### Cleanup
```bash
# Remove unused images
docker image prune

# Remove unused volumes
docker volume prune

# Complete cleanup
docker system prune -a
```

## Support

For issues and questions:
1. Check the logs first
2. Review this documentation
3. Check Docker and system requirements
4. Verify network connectivity and ports
