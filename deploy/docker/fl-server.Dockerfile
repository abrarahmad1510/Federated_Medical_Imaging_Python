# FL Server Dockerfile
FROM python:3.9-slim
WORKDIR /app
# Install system dependencies
RUN apt-get update && apt-get install -y \
gcc \
g++ \
libssl-dev \
curl \
&& rm -rf /var/lib/apt/lists/*
# Copy requirements
COPY requirements/base.txt .
RUN pip install --no-cache-dir -r base.txt
# Copy application code
COPY fl-server/ ./fl-server/
COPY clients/shared/ ./clients/shared/
COPY .env .
# Create non-root user
RUN useradd -m -u 1001 fluser && chown -R fluser:fluser /app
USER fluser
# Create necessary directories
RUN mkdir -p /app/logs /app/models /app/checkpoints /app/certs
# Expose ports
EXPOSE 8080 9091
# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
CMD curl -f http://localhost:8080 || exit 1
# Start command
CMD ["python", "fl-server/server.py"]
