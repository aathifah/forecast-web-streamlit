"""
Konfigurasi Keamanan untuk Aplikasi Forecasting
"""

import os
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# =============================================================================
# SECURITY CONFIGURATION - Streamlit Optimized
# =============================================================================
# Konfigurasi keamanan yang dioptimalkan untuk deployment Streamlit

import os
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Streamlit-specific security configuration
STREAMLIT_SECURITY_CONFIG = {
    # File handling
    "MAX_FILE_SIZE": 100 * 1024 * 1024,  # 100MB max file size
    "ALLOWED_EXTENSIONS": [".xlsx", ".xls"],
    "MAX_FILE_AGE_HOURS": 2,  # Delete files after 2 hours
    
    # Rate limiting - Streamlit optimized
    "RATE_LIMIT_REQUESTS": 100,  # Max 100 requests per minute (Streamlit local)
    "RATE_LIMIT_WINDOW": 60,  # 60 seconds
    
    # User management - Streamlit optimized
    "MAX_CONCURRENT_USERS": 1,  # Streamlit single user per session
    "USER_SESSION_TIMEOUT": 7200,  # 2 jam timeout (7200 detik)
    
    # Data validation
    "MIN_DATA_ROWS": 5,
    "MAX_DATA_ROWS": 1000000,  # Maksimal baris data (ditingkatkan untuk Railway)
    "REQUIRED_COLUMNS": ['PART_NO', 'MONTH', 'ORIGINAL_SHIPPING_QTY'],
    "VALIDATE_EXCEL_STRUCTURE": True,  # Enable Excel structure validation
    
    # Security features
    "ENCRYPT_TEMP_FILES": True,
    "AUTO_DELETE_AFTER_DOWNLOAD": True,
    "MEMORY_WIPE_AFTER_PROCESSING": True,
    "SECURE_FILE_DELETION": True,
    
    # Network security
    "ENFORCE_HTTPS": True,
    "SECURE_COOKIES": True,
    
    # Streamlit specific optimizations
    "ENABLE_LAZY_LOADING": True,
    "MEMORY_OPTIMIZATION": True,
    "TIMEOUT_OPTIMIZATION": True,
    "STREAMLIT_CACHING": True,
    "SESSION_STATE_MANAGEMENT": True
}

# Enhanced Security Headers untuk Streamlit
STREAMLIT_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "SAMEORIGIN",
    "X-XSS-Protection": "1; mode=block",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self';",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=(), payment=(), usb=(), magnetometer=(), gyroscope=()",
    "X-Permitted-Cross-Domain-Policies": "none",
    "X-Download-Options": "noopen",
    "X-DNS-Prefetch-Control": "off",
    "Cross-Origin-Embedder-Policy": "require-corp",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin"
}

def get_streamlit_security_config():
    """Get Streamlit-specific security configuration"""
    return STREAMLIT_SECURITY_CONFIG

def get_streamlit_security_headers():
    """Get Streamlit-specific security headers"""
    return STREAMLIT_SECURITY_HEADERS

def validate_streamlit_environment():
    """Validate Streamlit environment variables"""
    required_env_vars = [
        'STREAMLIT_SERVER_PORT',
        'STREAMLIT_SERVER_ADDRESS'
    ]
    
    missing_vars = []
    for var in required_env_vars:
        if not os.environ.get(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.warning(f"Missing Streamlit environment variables: {missing_vars}")
        return False
    
    logger.info("✅ Streamlit environment validation passed")
    return True

def setup_streamlit_optimizations():
    """Setup Streamlit-specific optimizations"""
    # Set environment variables for optimization
    os.environ.setdefault('PYTHONUNBUFFERED', '1')
    os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
    os.environ.setdefault('MKL_NUM_THREADS', '2')
    os.environ.setdefault('JOBLIB_TEMP_FOLDER', '/tmp')
    os.environ.setdefault('STREAMLIT_SERVER_HEADLESS', 'true')
    os.environ.setdefault('STREAMLIT_SERVER_ENABLE_CORS', 'false')
    
    logger.info("✅ Streamlit optimizations applied")

def get_streamlit_memory_limit():
    """Get Streamlit memory limit"""
    memory_limit = os.environ.get('MEMORY_LIMIT', '1G')
    logger.info(f"Streamlit memory limit: {memory_limit}")
    return memory_limit

def check_streamlit_deployment():
    """Check if running on Streamlit"""
    is_streamlit = bool(os.environ.get('STREAMLIT_SERVER_PORT'))
    if is_streamlit:
        logger.info("✅ Running on Streamlit platform")
    else:
        logger.info("⚠️ Not running on Streamlit platform")
    return is_streamlit 
