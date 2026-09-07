import os

# Keep the portfolio deployment lightweight. Override with WEB_CONCURRENCY
# when running on a larger production instance.
workers = int(os.getenv("WEB_CONCURRENCY", "1"))

bind = "0.0.0.0:8000"
worker_class = "uvicorn.workers.UvicornWorker"

timeout = 120
keepalive = 5

accesslog = "-"
errorlog = "-"
loglevel = "info"

preload_app = True
