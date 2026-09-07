import os

# Render provides PORT for web services. Keep 10000 as the local/default
# fallback while allowing WEB_CONCURRENCY to tune worker count.
workers = int(os.getenv("WEB_CONCURRENCY", "1"))

bind = f"0.0.0.0:{os.getenv('PORT', '10000')}"
worker_class = "uvicorn.workers.UvicornWorker"

timeout = 120
keepalive = 5

accesslog = "-"
errorlog = "-"
loglevel = "info"

preload_app = True
