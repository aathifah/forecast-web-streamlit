# Railway-optimized Dockerfile untuk forecast service
FROM python:3.11-slim

# Set Railway-specific environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

# Set work directory
WORKDIR /app

# Install system dependencies untuk Railway
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libopenblas-dev \
    liblapack-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies dengan Railway optimizations
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --upgrade setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories untuk Railway
RUN mkdir -p /tmp/forecast_results && \
    mkdir -p /tmp/forecast_cache && \
    mkdir -p static && \
    chmod 755 /tmp/forecast_results && \
    chmod 755 /tmp/forecast_cache && \
    chown -R 1000:1000 /app && \
    chmod -R 755 /app

# Expose port
EXPOSE 8080

# Health check untuk Railway - optimized untuk faster startup
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Railway-optimized startup command dengan timeout handling
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1 --timeout-keep-alive 300 --timeout-graceful-shutdown 30 --access-log --log-level warning"]
