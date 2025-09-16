# gunicorn_config.py
import os
import multiprocessing

# Server socket
bind = "127.0.0.1:8001"
backlog = 2048

# Worker processes
workers = min(2, multiprocessing.cpu_count())
worker_class = "sync"
worker_connections = 1000
timeout = 300
keepalive = 2

# Memory management
max_requests = 1000
max_requests_jitter = 50
preload_app = True

# Logging
accesslog = "/var/log/iph-forecasting/gunicorn_access.log"
errorlog = "/var/log/iph-forecasting/gunicorn_error.log"
loglevel = "info"

# Process naming
proc_name = 'iph-forecasting'

# Server mechanics
daemon = False
pidfile = "/var/log/iph-forecasting/gunicorn.pid"
user = "www-data"
group = "www-data"

# Graceful timeout
graceful_timeout = 30

def when_ready(server):
    server.log.info("IPH Forecasting Server ready at https://iph.bpskotabatu.com")