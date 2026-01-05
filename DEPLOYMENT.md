# Jo.E Deployment Guide

## Local Deployment

### Standard Setup

1. **Clone Repository**
```bash
git clone https://github.com/HimJoe/Joint-Evaluations-Jo.E-.git
cd Joint-Evaluations-Jo.E-
```

2. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Run Application**
```bash
streamlit run app.py
```

5. **Access Application**
- Open browser to: http://localhost:8501
- Default port: 8501
- Custom port: `streamlit run app.py --server.port 8080`

## Cloud Deployment

### Streamlit Community Cloud

1. **Push to GitHub**
```bash
git push origin main
```

2. **Deploy on Streamlit Cloud**
- Visit: https://share.streamlit.io
- Click "New app"
- Select repository: `HimJoe/Joint-Evaluations-Jo.E-`
- Main file: `app.py`
- Click "Deploy"

3. **Configure Secrets** (if using APIs)
- Go to App Settings → Secrets
- Add API keys:
```toml
[openai]
api_key = "your-openai-key"

[anthropic]
api_key = "your-anthropic-key"
```

### Heroku

1. **Create Procfile**
```bash
echo "web: streamlit run app.py --server.port \$PORT --server.address 0.0.0.0" > Procfile
```

2. **Create runtime.txt**
```bash
echo "python-3.11.6" > runtime.txt
```

3. **Deploy**
```bash
heroku create joe-evaluation-app
git push heroku main
heroku open
```

### AWS EC2

1. **Launch EC2 Instance**
- AMI: Ubuntu 22.04 LTS
- Instance type: t2.medium (minimum)
- Security group: Allow inbound on port 8501

2. **Connect and Setup**
```bash
ssh -i your-key.pem ubuntu@your-instance-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python
sudo apt install python3.11 python3-pip -y

# Clone repository
git clone https://github.com/HimJoe/Joint-Evaluations-Jo.E-.git
cd Joint-Evaluations-Jo.E-

# Install dependencies
pip3 install -r requirements.txt

# Run with nohup
nohup streamlit run app.py --server.port 8501 &
```

3. **Access Application**
- http://your-instance-ip:8501

### Docker Deployment

1. **Create Dockerfile**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
```

2. **Build and Run**
```bash
# Build image
docker build -t joe-evaluation .

# Run container
docker run -p 8501:8501 joe-evaluation

# Or with docker-compose
docker-compose up
```

3. **Docker Compose** (create docker-compose.yml)
```yaml
version: '3.8'

services:
  joe-app:
    build: .
    ports:
      - "8501:8501"
    environment:
      - STREAMLIT_SERVER_PORT=8501
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
    volumes:
      - ./data:/app/data
    restart: unless-stopped
```

### Kubernetes

1. **Create Deployment** (k8s-deployment.yaml)
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: joe-evaluation
spec:
  replicas: 3
  selector:
    matchLabels:
      app: joe-evaluation
  template:
    metadata:
      labels:
        app: joe-evaluation
    spec:
      containers:
      - name: joe-app
        image: your-registry/joe-evaluation:latest
        ports:
        - containerPort: 8501
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
---
apiVersion: v1
kind: Service
metadata:
  name: joe-evaluation-service
spec:
  selector:
    app: joe-evaluation
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8501
  type: LoadBalancer
```

2. **Deploy to Kubernetes**
```bash
kubectl apply -f k8s-deployment.yaml
kubectl get services  # Get external IP
```

## Production Considerations

### Performance Optimization

1. **Enable Caching**
```python
# In app.py, use Streamlit caching
@st.cache_data
def load_data():
    # Expensive operations
    pass

@st.cache_resource
def load_model():
    # Model loading
    pass
```

2. **Optimize Resource Usage**
```bash
# Limit memory in Streamlit config
mkdir -p ~/.streamlit
cat > ~/.streamlit/config.toml << EOF
[server]
maxUploadSize = 50
maxMessageSize = 50

[browser]
gatherUsageStats = false
EOF
```

### Security

1. **Enable HTTPS**
```bash
# Using nginx as reverse proxy
sudo apt install nginx

# Configure nginx
sudo nano /etc/nginx/sites-available/joe-evaluation

# Add SSL certificate (Let's Encrypt)
sudo certbot --nginx -d your-domain.com
```

2. **Set Environment Variables**
```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export SECRET_KEY="your-secret"
```

3. **Authentication** (if needed)
```python
# Add to app.py
import streamlit_authenticator as stauth

authenticator = stauth.Authenticate(
    credentials,
    'joe_evaluation',
    'auth_key',
    cookie_expiry_days=30
)

name, authentication_status, username = authenticator.login('Login', 'main')
```

### Monitoring

1. **Application Logs**
```bash
# View logs
streamlit run app.py 2>&1 | tee app.log

# Monitor with tail
tail -f app.log
```

2. **Health Checks**
```python
# Add health endpoint
@st.cache_data(ttl=60)
def health_check():
    return {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    }
```

3. **Performance Monitoring**
- Use Streamlit built-in metrics
- Set up external monitoring (Datadog, New Relic)
- Monitor API rate limits

### Scaling

1. **Horizontal Scaling**
- Deploy multiple instances behind load balancer
- Use Kubernetes with HPA (Horizontal Pod Autoscaler)
- Configure session affinity if needed

2. **Database for State Management**
```python
# Replace session_state with database
import psycopg2

def save_evaluation(evaluation_data):
    conn = psycopg2.connect(DATABASE_URL)
    # Save to database
    conn.close()
```

3. **Caching Layer**
```python
# Use Redis for shared cache
import redis

cache = redis.Redis(host='localhost', port=6379)
```

## Environment-Specific Configurations

### Development
```toml
# .streamlit/config.toml
[server]
runOnSave = true
port = 8501

[logger]
level = "debug"
```

### Staging
```toml
[server]
runOnSave = false
port = 8501

[logger]
level = "info"
```

### Production
```toml
[server]
runOnSave = false
enableCORS = false
enableXsrfProtection = true

[logger]
level = "warning"

[client]
showErrorDetails = false
```

## Backup and Recovery

1. **Backup Evaluation Data**
```bash
# If using database
pg_dump joe_evaluations > backup_$(date +%Y%m%d).sql

# If using file storage
tar -czf evaluations_backup.tar.gz data/
```

2. **Automated Backups**
```bash
# Add to crontab
0 2 * * * /path/to/backup_script.sh
```

## Troubleshooting

### Common Issues

1. **Port Already in Use**
```bash
# Find process using port 8501
lsof -i :8501
# Kill process
kill -9 <PID>
```

2. **Memory Issues**
```bash
# Increase swap space (Linux)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

3. **Dependency Conflicts**
```bash
# Create fresh virtual environment
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Maintenance

### Regular Updates

1. **Update Dependencies**
```bash
pip install --upgrade -r requirements.txt
pip freeze > requirements.txt
```

2. **Security Patches**
```bash
pip install --upgrade pip
pip audit  # Check for vulnerabilities
```

3. **Streamlit Updates**
```bash
pip install --upgrade streamlit
streamlit cache clear
```

### Monitoring Checklist

- [ ] Application uptime
- [ ] Response times
- [ ] Error rates
- [ ] Memory usage
- [ ] CPU utilization
- [ ] API quota usage
- [ ] Disk space
- [ ] SSL certificate expiry

## Support

For deployment issues:
- GitHub Issues: https://github.com/HimJoe/Joint-Evaluations-Jo.E-/issues
- Documentation: Check APP_README.md
- Streamlit Docs: https://docs.streamlit.io/

---

**Last Updated**: January 2026
