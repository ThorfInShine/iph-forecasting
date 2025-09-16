# gunicorn_config.py
import os
import multiprocessing

# Server socket
bind = "127.0.0.1:8000"
backlog = 2048

# Worker processes
workers = min(4, (multiprocessing.cpu_count() * 2) + 1)
worker_class = "sync"
worker_connections = 1000
timeout = 120
keepalive = 2

# Restart workers after this many requests
max_requests = 1000
max_requests_jitter = 50

# Logging
accesslog = "/var/log/gunicorn/access.log"
errorlog = "/var/log/gunicorn/error.log"
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