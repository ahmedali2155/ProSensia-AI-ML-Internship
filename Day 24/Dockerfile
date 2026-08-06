# ---------------------------------------------------------
# Dockerfile - Titanic Survival Prediction AI Microservice
#
# Virtual Machines vs Docker Containers
# - Virtual Machines virtualize the entire operating system,
#   requiring a hypervisor and a full guest OS.
# - Docker containers share the host operating system kernel,
#   making them lightweight, faster to start, and more efficient.
#
# Docker Layer Caching
# - Docker caches layers such as dependency installation.
# - If only the application code changes, Docker reuses the
#   cached dependency layers, resulting in much faster rebuilds.
# ---------------------------------------------------------

# Use lightweight Python image
FROM python:3.11-slim

# Create working directory
WORKDIR /app

# Copy requirements first (better Docker layer caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create non-root user
RUN useradd -m appuser

# Switch to non-root user
USER appuser

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]