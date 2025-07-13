# Cloud Deployment Guide

This document provides instructions for deploying the Premier League MLOps system to various cloud platforms.

## Deployment Options

### 1. Railway (Recommended)

Railway provides the simplest deployment experience with automatic PostgreSQL database provisioning.

#### Steps:
1. **Connect Repository**:
   - Go to [Railway](https://railway.app)
   - Connect your GitHub repository
   - Select the `main` branch

2. **Add PostgreSQL**:
   - In Railway dashboard, click "Add Service"
   - Select "PostgreSQL"
   - Railway will automatically set environment variables

3. **Deploy**:
   - Railway will automatically detect the `railway.toml` configuration
   - The application will be deployed with health checks
   - Access your app at the provided Railway URL

#### Configuration:
- Health check endpoint: `/health`
- Automatic restarts on failure
- PostgreSQL database included
- Environment variables auto-configured

### 2. Render

Render offers a free tier with Docker support and managed PostgreSQL.

#### Steps:
1. **Connect Repository**:
   - Go to [Render](https://render.com)
   - Connect your GitHub repository
   - Select "Web Service"

2. **Configure Service**:
   - Build Command: `docker build -t app .`
   - Start Command: `docker run -p $PORT:8000 app`
   - Or use the provided `render.yaml` blueprint

3. **Add Database**:
   - Create a PostgreSQL database service
   - Connect to your web service via environment variables

### 3. Fly.io

Fly.io provides global deployment with Docker support.

#### Steps:
1. **Install Fly CLI**:
   ```bash
   curl -L https://fly.io/install.sh | sh
   ```

2. **Initialize App**:
   ```bash
   fly launch
   ```

3. **Deploy**:
   ```bash
   fly deploy
   ```

## Environment Variables

All cloud deployments require these environment variables:

### Required:
- `POSTGRES_HOST` - Database host
- `POSTGRES_PORT` - Database port (usually 5432)
- `POSTGRES_DB` - Database name
- `POSTGRES_USER` - Database user
- `POSTGRES_PASSWORD` - Database password

### Optional:
- `MLFLOW_TRACKING_URI` - MLflow tracking server (defaults to SQLite)
- `MODEL_REGISTRATION_THRESHOLD` - Model accuracy threshold (default: 0.6)
- `PYTHONPATH` - Python path (default: /app)
- `PORT` - Application port (default: 8000)

## Health Checks

All deployments include health check endpoints:
- **Endpoint**: `/health`
- **Response**: JSON with system status
- **Timeout**: 30 seconds
- **Interval**: 30 seconds

## Monitoring

The deployed application includes:
- **API Metrics**: Request/response times, error rates
- **Model Metrics**: Prediction accuracy, drift detection
- **System Metrics**: Memory usage, CPU utilization

## Scaling

### Horizontal Scaling:
- Railway: Auto-scaling based on traffic
- Render: Manual scaling in dashboard
- Fly.io: Configure in `fly.toml`

### Database Scaling:
- Start with free tier PostgreSQL
- Upgrade to paid plans for production workloads
- Consider read replicas for high-traffic scenarios

## Security

### Production Considerations:
- Use strong database passwords
- Enable SSL/TLS for database connections
- Set up monitoring and alerting
- Regular security updates

### Environment Variables:
- Never commit secrets to version control
- Use platform-specific secret management
- Rotate credentials regularly

## Cost Optimization

### Free Tier Limits:
- **Railway**: 500 hours/month, 1GB RAM
- **Render**: 750 hours/month, 512MB RAM
- **Fly.io**: 160GB-hours/month

### Optimization Tips:
- Use efficient Docker images
- Implement proper caching
- Monitor resource usage
- Scale down during low traffic

## Troubleshooting

### Common Issues:

1. **Database Connection Errors**:
   - Check environment variables
   - Verify database service is running
   - Test connection strings

2. **Memory Issues**:
   - Optimize Docker image size
   - Monitor memory usage
   - Consider upgrading plan

3. **Build Failures**:
   - Check Dockerfile syntax
   - Verify dependency versions
   - Review build logs

### Support:
- Check platform documentation
- Review application logs
- Monitor health check endpoints
- Use platform support channels
