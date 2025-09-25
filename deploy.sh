#!/bin/bash

# OMR Scraper Deployment Script for Digital Ocean
# Run this script on your Digital Ocean droplet

set -e

echo "🚀 Starting OMR Scraper deployment..."

# Configuration
APP_NAME="omr-scraper"
APP_DIR="/var/www/$APP_NAME"
SERVICE_NAME="omr-scraper"
NGINX_SITE="omr-scraper"
DOMAIN="omr.yourdomain.com"  # Change this to your actual domain

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    print_error "Please run as root (use sudo)"
    exit 1
fi

# Update system packages
print_status "Updating system packages..."
apt update && apt upgrade -y

# Install required packages
print_status "Installing required packages..."
apt install -y python3 python3-pip python3-venv nginx ufw fail2ban

# Create application directory
print_status "Creating application directory..."
mkdir -p $APP_DIR
cd $APP_DIR

# Copy application files (assuming you've uploaded them)
print_status "Setting up application files..."
# Note: You need to upload your project files to this directory first

# Create virtual environment
print_status "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
print_status "Installing Python dependencies..."
pip install --upgrade pip
pip install opencv-python numpy Flask schedule gunicorn

# Set proper permissions
print_status "Setting up permissions..."
chown -R www-data:www-data $APP_DIR
chmod -R 755 $APP_DIR

# Create necessary directories
mkdir -p $APP_DIR/marked_images
mkdir -p $APP_DIR/logs
mkdir -p /tmp/omr_uploads
chown -R www-data:www-data $APP_DIR/marked_images
chown -R www-data:www-data $APP_DIR/logs
chown -R www-data:www-data /tmp/omr_uploads

# Setup systemd service
print_status "Setting up systemd service..."
cp $APP_DIR/omr-scraper.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable $SERVICE_NAME

# Setup nginx
print_status "Configuring nginx..."
cp $APP_DIR/nginx-omr-scraper.conf /etc/nginx/sites-available/$NGINX_SITE

# Replace domain placeholder
sed -i "s/omr.yourdomain.com/$DOMAIN/g" /etc/nginx/sites-available/$NGINX_SITE

# Enable site
ln -sf /etc/nginx/sites-available/$NGINX_SITE /etc/nginx/sites-enabled/

# Test nginx configuration
nginx -t

# Setup firewall
print_status "Configuring firewall..."
ufw allow ssh
ufw allow 'Nginx Full'
ufw --force enable

# Start services
print_status "Starting services..."
systemctl start $SERVICE_NAME
systemctl restart nginx

# Check service status
print_status "Checking service status..."
systemctl status $SERVICE_NAME --no-pager

print_status "🎉 Deployment completed successfully!"
print_status "Your OMR Scraper is now running at: http://$DOMAIN"
print_status "API endpoint: http://$DOMAIN/analyze-omr"

# Optional: Setup SSL with Let's Encrypt
read -p "Do you want to setup SSL with Let's Encrypt? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Setting up SSL with Let's Encrypt..."
    apt install -y certbot python3-certbot-nginx
    certbot --nginx -d $DOMAIN
    print_status "SSL setup completed!"
fi

print_status "Deployment finished! 🚀"
print_status "Service logs: journalctl -u $SERVICE_NAME -f"
print_status "Nginx logs: tail -f /var/log/nginx/omr-scraper.access.log"
