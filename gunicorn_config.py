# gunicorn_config.py
import os
import multiprocessing

# Server socket
bind = "127.0.0.1:8001"  # Port yang berbeda dari tutorial
backlog = 2048

# Worker processes
workers = min(2, (multiprocessing.cpu_count() * 2) + 1)  # Lebih konservatif untuk shared hosting
worker_class = "sync"
worker_connections = 1000
timeout = 300  # Timeout lebih lama untuk processing
keepalive = 2

# Restart workers after this many requests
max_requests = 1000
max_requests_jitter = 50

# Logging
accesslog = "/var/log/iph-forecasting/access.log"
errorlog = "/var/log/iph-forecasting/error.log"
loglevel = "info"

# Process naming
proc_name = 'iph-forecasting'

# Server mechanics
preload_app = True
daemon = False
user = "www-data"
group = "www-data"

# Application
wsgi_module = "app:app"

# Create log directories
import os
os.makedirs("/var/log/iph-forecasting", exist_ok=True)