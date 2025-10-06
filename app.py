#!/usr/bin/env python3
"""
Streamlit Web Forecast Application
Aplikasi forecasting dengan interface Streamlit yang user-friendly
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import tempfile
import shutil

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import modules
try:
    from forecast_service import process_forecast
    from kategoriparts_web import process_categorization
    from activity_monitor import get_monitor
    from security_config import get_streamlit_security_config, setup_streamlit_optimizations
except ImportError as e:
    st.error(f"❌ Error importing modules: {e}")
    st.stop()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup Streamlit optimizations
setup_streamlit_optimizations()

# Get security config
security_config = get_streamlit_security_config()

# Initialize activity monitor
monitor = get_monitor()

# Streamlit page config
st.set_page_config(
    page_title="Forecast Web Application",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-message {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.375rem;
        margin: 1rem 0;
    }
    .error-message {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.375rem;
        margin: 1rem 0;
    }
    .info-message {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.375rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def validate_uploaded_file(uploaded_file) -> bool:
    """Validate uploaded file"""
    if uploaded_file is None:
        return False
    
    # Check file extension
    allowed_extensions = security_config["ALLOWED_EXTENSIONS"]
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    
    if file_extension not in allowed_extensions:
        st.error(f"❌ File extension tidak didukung. Ekstensi yang diizinkan: {allowed_extensions}")
        return False
    
    # Check file size
    max_size = security_config["MAX_FILE_SIZE"]
    if uploaded_file.size > max_size:
        st.error(f"❌ File terlalu besar. Maksimal ukuran: {max_size // (1024*1024)}MB")
        return False
    
    return True

def process_file_upload(uploaded_file) -> pd.DataFrame:
    """Process uploaded file and return DataFrame"""
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_file_path = tmp_file.name
        
        # Read Excel file
        df = pd.read_excel(tmp_file_path)
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")
        return None

def display_forecast_results(forecast_data: Dict[str, Any]):
    """Display forecast results in Streamlit"""
    if not forecast_data or forecast_data.get('status') != 'success':
        st.error("❌ Tidak ada data forecast untuk ditampilkan")
        return
    
    data = forecast_data.get('data', {})
    
    # Display summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Backtest", data.get('total_backtest', 0))
    
    with col2:
        st.metric("Total Realtime", data.get('total_realtime', 0))
    
    with col3:
        avg_error = data.get('average_error', 0)
        st.metric("Average Error", f"{avg_error:.2f}%")
    
    with col4:
        processing_time = data.get('processing_time', 0)
        st.metric("Processing Time", f"{processing_time:.2f}s")
    
    # Display detailed results
    if 'backtest_results' in data:
        st.subheader("📊 Backtest Results")
        backtest_df = pd.DataFrame(data['backtest_results'])
        st.dataframe(backtest_df, use_container_width=True)
    
    if 'realtime_results' in data:
        st.subheader("🔮 Realtime Forecast Results")
        realtime_df = pd.DataFrame(data['realtime_results'])
        st.dataframe(realtime_df, use_container_width=True)

def display_categorization_results(categorization_data: Dict[str, Any]):
    """Display categorization results in Streamlit"""
    if not categorization_data or categorization_data.get('status') != 'success':
        st.error("❌ Tidak ada data kategorisasi untuk ditampilkan")
        return
    
    data = categorization_data.get('data', {})
    
    # Display summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Parts", data.get('total_parts', 0))
    
    with col2:
        st.metric("Smooth", data.get('smooth_count', 0))
    
    with col3:
        st.metric("Erratic", data.get('erratic_count', 0))
    
    with col4:
        st.metric("Intermittent", data.get('intermittent_count', 0))
    
    # Display detailed results
    if 'categorized_data' in data:
        st.subheader("📋 Categorized Parts")
        categorized_df = pd.DataFrame(data['categorized_data'])
        st.dataframe(categorized_df, use_container_width=True)

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">📊 Forecast Web Application</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Processing options
        st.subheader("Processing Options")
        max_data_months = st.slider("Max Data Months", 12, 48, 24, help="Maximum months of historical data to use")
        batch_size = st.slider("Batch Size", 5, 20, 10, help="Number of parts to process in each batch")
        
        # Model options
        st.subheader("Model Options")
        enable_lstm = st.checkbox("Enable LSTM", value=True, help="Enable LSTM neural network model")
        enable_ensemble = st.checkbox("Enable Ensemble", value=True, help="Enable ensemble model selection")
        
        # Security options
        st.subheader("Security Options")
        auto_cleanup = st.checkbox("Auto Cleanup", value=True, help="Automatically clean up temporary files")
        memory_limit = st.slider("Memory Limit (%)", 50, 90, 75, help="Memory usage limit before cleanup")
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Upload & Process", "📊 Forecast Results", "📋 Categorization Results", "📈 Monitoring"])
    
    with tab1:
        st.header("📁 Upload & Process Data")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose an Excel file",
            type=['xlsx', 'xls'],
            help="Upload your dataset in Excel format"
        )
        
        if uploaded_file is not None:
            # Validate file
            if validate_uploaded_file(uploaded_file):
                st.success("✅ File validated successfully")
                
                # Process file
                with st.spinner("Processing file..."):
                    df = process_file_upload(uploaded_file)
                    
                    if df is not None:
                        st.success(f"✅ File processed successfully. Shape: {df.shape}")
                        
                        # Display data preview
                        st.subheader("📋 Data Preview")
                        st.dataframe(df.head(10), use_container_width=True)
                        
                        # Store in session state
                        st.session_state['uploaded_data'] = df
                        st.session_state['file_name'] = uploaded_file.name
                        
                        # Processing buttons
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if st.button("🔮 Run Forecast", type="primary", use_container_width=True):
                                if 'uploaded_data' in st.session_state:
                                    with st.spinner("Running forecast..."):
                                        try:
                                            # Log activity
                                            monitor.log_streamlit_activity(
                                                session_id=st.session_state.get('session_id', 'default'),
                                                activity_type="forecast",
                                                file_uploaded=True,
                                                file_size=uploaded_file.size,
                                                processing_time=0.0,
                                                status="started"
                                            )
                                            
                                            # Run forecast
                                            forecast_results = process_forecast(
                                                st.session_state['uploaded_data'],
                                                max_data_months=max_data_months,
                                                batch_size=batch_size,
                                                enable_lstm=enable_lstm,
                                                enable_ensemble=enable_ensemble
                                            )
                                            
                                            # Store results
                                            st.session_state['forecast_results'] = forecast_results
                                            
                                            # Log completion
                                            monitor.log_streamlit_activity(
                                                session_id=st.session_state.get('session_id', 'default'),
                                                activity_type="forecast",
                                                file_uploaded=True,
                                                file_size=uploaded_file.size,
                                                processing_time=forecast_results.get('processing_time', 0),
                                                status="completed"
                                            )
                                            
                                            st.success("✅ Forecast completed successfully!")
                                            st.rerun()
                                            
                                        except Exception as e:
                                            st.error(f"❌ Forecast failed: {str(e)}")
                                            monitor.log_streamlit_activity(
                                                session_id=st.session_state.get('session_id', 'default'),
                                                activity_type="forecast",
                                                file_uploaded=True,
                                                file_size=uploaded_file.size,
                                                processing_time=0.0,
                                                status="failed"
                                            )
                        
                        with col2:
                            if st.button("📋 Run Categorization", type="secondary", use_container_width=True):
                                if 'uploaded_data' in st.session_state:
                                    with st.spinner("Running categorization..."):
                                        try:
                                            # Log activity
                                            monitor.log_streamlit_activity(
                                                session_id=st.session_state.get('session_id', 'default'),
                                                activity_type="categorization",
                                                file_uploaded=True,
                                                file_size=uploaded_file.size,
                                                processing_time=0.0,
                                                status="started"
                                            )
                                            
                                            # Run categorization
                                            categorization_results = process_categorization(
                                                st.session_state['uploaded_data'],
                                                max_data_months=max_data_months
                                            )
                                            
                                            # Store results
                                            st.session_state['categorization_results'] = categorization_results
                                            
                                            # Log completion
                                            monitor.log_streamlit_activity(
                                                session_id=st.session_state.get('session_id', 'default'),
                                                activity_type="categorization",
                                                file_uploaded=True,
                                                file_size=uploaded_file.size,
                                                processing_time=categorization_results.get('processing_time', 0),
                                                status="completed"
                                            )
                                            
                                            st.success("✅ Categorization completed successfully!")
                                            st.rerun()
                                            
                                        except Exception as e:
                                            st.error(f"❌ Categorization failed: {str(e)}")
                                            monitor.log_streamlit_activity(
                                                session_id=st.session_state.get('session_id', 'default'),
                                                activity_type="categorization",
                                                file_uploaded=True,
                                                file_size=uploaded_file.size,
                                                processing_time=0.0,
                                                status="failed"
                                            )
    
    with tab2:
        st.header("📊 Forecast Results")
        
        if 'forecast_results' in st.session_state:
            display_forecast_results(st.session_state['forecast_results'])
        else:
            st.info("ℹ️ No forecast results available. Please run forecast first.")
    
    with tab3:
        st.header("📋 Categorization Results")
        
        if 'categorization_results' in st.session_state:
            display_categorization_results(st.session_state['categorization_results'])
        else:
            st.info("ℹ️ No categorization results available. Please run categorization first.")
    
    with tab4:
        st.header("📈 System Monitoring")
        
        # Display monitoring information
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💾 Memory Usage")
            memory_summary = monitor.get_system_summary()
            if memory_summary.get('avg_memory_usage_mb'):
                st.metric("Average Memory Usage", f"{memory_summary['avg_memory_usage_mb']:.2f} MB")
        
        with col2:
            st.subheader("🖥️ CPU Usage")
            if memory_summary.get('avg_cpu_usage_percent'):
                st.metric("Average CPU Usage", f"{memory_summary['avg_cpu_usage_percent']:.2f}%")
        
        # Display activity summary
        st.subheader("📊 Activity Summary")
        activity_summary = monitor.get_full_summary()
        
        if activity_summary.get('forecast', {}).get('total_parts_processed', 0) > 0:
            st.metric("Parts Processed", activity_summary['forecast']['total_parts_processed'])
            st.metric("Average Processing Time", f"{activity_summary['forecast']['avg_processing_time']:.2f}s")
        
        # Display recent activities
        st.subheader("🕒 Recent Activities")
        if activity_summary.get('web_activity', {}).get('recent_activities'):
            activities_df = pd.DataFrame(activity_summary['web_activity']['recent_activities'])
            st.dataframe(activities_df, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("**Forecast Web Application** - Powered by Streamlit")
    st.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    # Initialize session state
    if 'session_id' not in st.session_state:
        st.session_state['session_id'] = f"session_{int(time.time())}"
    
    # Run main application
    main()
