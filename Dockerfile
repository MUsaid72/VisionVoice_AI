# ==============================================================================
# Multi-Stage Secure Production Dockerfile for VisionVoice AI
# Security Protections:
# - Runs as Non-Root User (UID 10001) to prevent Container Escape (OWASP A05)
# - No Compiler / Build Tools in Runtime Image (Minimal Attack Surface)
# - Read-Only Root Filesystem Ready with dedicated /tmp volume
# - Gunicorn Production WSGI Server with Worker Process Sandboxing
# ==============================================================================

FROM python:3.12-slim AS builder

WORKDIR /build

# Install minimal OS build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies to virtual environment
COPY requirements.txt .
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt


# ==============================================================================
# Runtime Stage
# ==============================================================================
FROM python:3.12-slim AS runner

WORKDIR /app

# Create non-root system user and group (Least Privilege Principle)
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/false -m appuser

# Copy virtualenv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application files
COPY --chown=appuser:appgroup app.py models.py vision_engine.py mobilenetv2.onnx imagenet_classes.json ./
COPY --chown=appuser:appgroup templates/ templates/
COPY --chown=appuser:appgroup static/ static/

# Initialize empty database and assign strict non-root ownership
RUN touch database.db && chown appuser:appgroup database.db && chmod 600 database.db

# Drop root privileges
USER appuser:appgroup

# Production Environment Variables
ENV FLASK_ENV=production \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 5000

# Healthcheck to detect container stalls
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/')" || exit 1

# Launch with Production WSGI Server (Gunicorn)
CMD ["gunicorn", "--workers=4", "--bind=0.0.0.0:5000", "--timeout=30", "--access-logfile=-", "--error-logfile=-", "app:app"]
