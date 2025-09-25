# 🚀 OMR Scraper Digital Ocean Deployment Guide

এই guide টি আপনাকে Digital Ocean droplet এ OMR Scraper deploy করতে সাহায্য করবে।

## 📋 Prerequisites

- Digital Ocean droplet (Ubuntu 20.04+ recommended)
- Root access বা sudo privileges
- Domain name configured (optional but recommended)
- Basic knowledge of Linux commands

## 🔧 Step-by-Step Deployment

### 1. Server Preparation

```bash
# Connect to your Digital Ocean droplet
ssh root@your-droplet-ip

# Update system
apt update && apt upgrade -y

# Install essential packages
apt install -y python3 python3-pip python3-venv nginx ufw fail2ban git
```

### 2. Upload Project Files

#### Option A: Using SCP (from your local machine)
```bash
# From your local machine, upload the project
scp -r /home/devzkhalil/Downloads/python-omr-scraper-main root@your-droplet-ip:/var/www/
```

#### Option B: Using Git (if you have a repository)
```bash
# On the server
cd /var/www/
git clone your-repository-url omr-scraper
```

### 3. Run Deployment Script

```bash
# Make script executable and run
cd /var/www/omr-scraper
chmod +x deploy.sh
sudo ./deploy.sh
```

### 4. Manual Configuration (if needed)

#### Configure Domain
```bash
# Edit nginx configuration
nano /etc/nginx/sites-available/omr-scraper

# Replace 'omr.yourdomain.com' with your actual domain
# Save and exit
```

#### Update Service Configuration
```bash
# Edit service file
nano /etc/systemd/system/omr-scraper.service

# Update the domain and secret key
# Save and exit
```

### 5. Start Services

```bash
# Reload systemd and start services
systemctl daemon-reload
systemctl enable omr-scraper
systemctl start omr-scraper
systemctl restart nginx

# Check status
systemctl status omr-scraper
```

## 🌐 Domain Configuration

### DNS Setup
1. Go to your domain registrar
2. Add an A record:
   - **Name**: `omr` (or your preferred subdomain)
   - **Type**: A
   - **Value**: Your Digital Ocean droplet IP
   - **TTL**: 300 (5 minutes)

### SSL Certificate (Optional but Recommended)
```bash
# Install Certbot
apt install -y certbot python3-certbot-nginx

# Get SSL certificate
certbot --nginx -d omr.yourdomain.com

# Auto-renewal (already configured by certbot)
```

## 🔍 Verification

### Check Service Status
```bash
# Check if service is running
systemctl status omr-scraper

# Check logs
journalctl -u omr-scraper -f

# Check nginx status
systemctl status nginx
```

### Test API Endpoints
```bash
# Test home endpoint
curl http://your-domain.com/

# Test health endpoint
curl http://your-domain.com/health
```

## 📊 Monitoring and Maintenance

### View Logs
```bash
# Application logs
journalctl -u omr-scraper -f

# Nginx logs
tail -f /var/log/nginx/omr-scraper.access.log
tail -f /var/log/nginx/omr-scraper.error.log

# Application logs
tail -f /var/log/omr_scraper.log
```

### Restart Services
```bash
# Restart OMR scraper
systemctl restart omr-scraper

# Restart nginx
systemctl restart nginx

# Restart both
systemctl restart omr-scraper nginx
```

### Update Application
```bash
# Stop service
systemctl stop omr-scraper

# Update code (if using git)
cd /var/www/omr-scraper
git pull origin main

# Update dependencies
source venv/bin/activate
pip install -r requirements.txt

# Start service
systemctl start omr-scraper
```

## 🛡️ Security Considerations

### Firewall Configuration
```bash
# Check firewall status
ufw status

# Allow only necessary ports
ufw allow ssh
ufw allow 'Nginx Full'
ufw enable
```

### Fail2ban Configuration
```bash
# Check fail2ban status
systemctl status fail2ban

# View banned IPs
fail2ban-client status nginx-http-auth
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Service Won't Start
```bash
# Check service logs
journalctl -u omr-scraper -n 50

# Check if port is in use
netstat -tlnp | grep 8001
```

#### 2. Nginx Configuration Issues
```bash
# Test nginx configuration
nginx -t

# Check nginx logs
tail -f /var/log/nginx/error.log
```

#### 3. Permission Issues
```bash
# Fix permissions
chown -R www-data:www-data /var/www/omr-scraper
chmod -R 755 /var/www/omr-scraper
```

#### 4. Port Already in Use
```bash
# Find process using port 8001
lsof -i :8001

# Kill process if needed
kill -9 <PID>
```

## 📈 Performance Optimization

### Nginx Optimization
```bash
# Edit nginx configuration
nano /etc/nginx/nginx.conf

# Add these settings in http block:
# worker_processes auto;
# worker_connections 1024;
# keepalive_timeout 65;
```

### System Optimization
```bash
# Increase file limits
echo "* soft nofile 65536" >> /etc/security/limits.conf
echo "* hard nofile 65536" >> /etc/security/limits.conf
```

## 🎯 API Usage Examples

### Upload and Analyze OMR
```bash
curl -X POST http://your-domain.com/analyze-omr \
  -F "image=@/path/to/omr-sheet.png" \
  -F "correct_answers={\"1\":\"3\",\"2\":\"1\",\"3\":\"4\"}"
```

### Check Service Status
```bash
curl http://your-domain.com/cleanup/status
```

## 📞 Support

If you encounter any issues:

1. Check the logs first
2. Verify all services are running
3. Check firewall and DNS settings
4. Ensure all dependencies are installed

## 🎉 Success!

Once deployed, your OMR Scraper will be available at:
- **URL**: `http://your-domain.com`
- **API**: `http://your-domain.com/analyze-omr`
- **Health Check**: `http://your-domain.com/health`

Your OMR scraper is now ready to process OMR sheets with Bengali support! 🚀
