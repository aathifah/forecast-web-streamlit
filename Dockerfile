# Streamlit-optimized Dockerfile untuk forecast application
FROM python:3.11-slim

# Set Streamlit-specific environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLE_CORS=false

# Set work directory
WORKDIR /app

# Install system dependencies untuk Streamlit
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libopenblas-dev \
    liblapack-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies dengan Streamlit optimizations
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --upgrade setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories untuk Streamlit
RUN mkdir -p /tmp/forecast_results && \
    mkdir -p /tmp/forecast_cache && \
    mkdir -p logs && \
    chmod 755 /tmp/forecast_results && \
    chmod 755 /tmp/forecast_cache && \
    chmod 755 logs && \
    chown -R 1000:1000 /app && \
    chmod -R 755 /app

# Expose port
EXPOSE 8501

# Health check untuk Streamlit
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Streamlit-optimized startup command
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
