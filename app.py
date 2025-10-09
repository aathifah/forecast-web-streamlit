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

# Custom CSS untuk UI seperti gambar
st.markdown("""
<style>
    /* Import font dari Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide default Streamlit elements */
    .stApp > header {
        display: none;
    }
    
    .stApp > div[data-testid="stToolbar"] {
        display: none;
    }
    
    /* Main container */
    .main-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
        padding: 2rem;
    }
    
    /* Welcome section */
    .welcome-section {
        text-align: center;
        margin-bottom: 3rem;
        color: white;
    }
    
    .welcome-title {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .welcome-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-bottom: 2rem;
    }
    
    /* Upload section */
    .upload-section {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem auto;
        max-width: 600px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
    }
    
    .upload-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #333;
        margin-bottom: 1rem;
    }
    
    .upload-instructions {
        color: #666;
        margin-bottom: 1.5rem;
        line-height: 1.6;
    }
    
    .upload-instructions strong {
        color: #333;
    }
    
    /* Dashboard containers */
    .dashboard-container {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 2rem;
        margin-top: 3rem;
    }
    
    .dashboard-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
    }
    
    .dashboard-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #333;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    /* Metrics cards */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
    }
    
    /* Chart containers */
    .chart-container {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        min-height: 200px;
    }
    
    /* Custom file uploader */
    .stFileUploader > div {
        background: #f8f9fa;
        border: 2px dashed #dee2e6;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div:hover {
        border-color: #667eea;
        background: #f0f2ff;
    }
    
    /* Custom buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Hide sidebar */
    .stSidebar {
        display: none;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .dashboard-container {
            grid-template-columns: 1fr;
        }
        
        .welcome-title {
            font-size: 2rem;
        }
        
        .upload-section {
            margin: 1rem;
            padding: 1.5rem;
        }
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

def create_upload_section():
    """Create upload section like in the image"""
    st.markdown("""
    <div class="upload-section">
        <div class="upload-title">Halo!</div>
        <div class="upload-instructions">
            Upload dataset (.xlsx) Anda di sini! (Max. 40 Part Number)<br><br>
            <strong>Dataset history harus dari satu depo saja dan memiliki kolom berikut:</strong><br>
            PART_NO, MONTH, ORIGINAL_SHIPPING_QTY atau ORDER_QTY<br><br>
            <strong>Kolom opsional:</strong><br>
            WORKING_DAYS, INVENTORY_CONTROL_CLASS
        </div>
    </div>
    """, unsafe_allow_html=True)
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose an Excel file",
            type=['xlsx', 'xls'],
        help="Upload your dataset in Excel format",
        label_visibility="collapsed"
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
                        
                        # Store in session state
                        st.session_state['uploaded_data'] = df
                        st.session_state['file_name'] = uploaded_file.name
                        
                        # Processing buttons
                        col1, col2 = st.columns(2)
                        
                        with col1:
                        if st.button("🔮 Forecast Excel File", type="primary", use_container_width=True):
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
                                            max_data_months=24,
                                            batch_size=10,
                                            enable_lstm=True,
                                            enable_ensemble=True
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
                                            max_data_months=24
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
    
def create_dashboard_prediction():
    """Create prediction dashboard like in the image"""
    st.markdown("""
    <div class="dashboard-card">
        <div class="dashboard-title">Prediksi Permintaan di Bulan Mendatang</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Get real data if available
    forecast_data = st.session_state.get('forecast_results', {})
    realtime_data = forecast_data.get('data', {}).get('realtime_results', [])
    
    # Default values
    icc_value = "A1"
    forecast_2025_01 = "1,316,571"
    forecast_2025_02 = "1,358,000"
    
    # Update with real data if available
    if realtime_data:
        # Get the first part's forecast for display
        first_part = realtime_data[0] if realtime_data else {}
        if 'forecast_2025_01' in first_part:
            forecast_2025_01 = f"{first_part['forecast_2025_01']:,.0f}"
        if 'forecast_2025_02' in first_part:
            forecast_2025_02 = f"{first_part['forecast_2025_02']:,.0f}"
        if 'inventory_control_class' in first_part:
            icc_value = first_part['inventory_control_class']
    
    # Metrics cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">ICC</div>
            <div class="metric-value">{icc_value}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Forecast 2025-01</div>
            <div class="metric-value">{forecast_2025_01}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Forecast 2025-02</div>
            <div class="metric-value">{forecast_2025_02}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart untuk forecast
        # Get real forecast data if available
        forecast_values = [1316571, 1358000, 1280000, 1420000]  # Default values
        if realtime_data:
            # Extract forecast values from real data
            first_part = realtime_data[0] if realtime_data else {}
            forecast_values = [
                first_part.get('forecast_2025_01', 1316571),
                first_part.get('forecast_2025_02', 1358000),
                first_part.get('forecast_2025_03', 1280000),
                first_part.get('forecast_2025_04', 1420000)
            ]
        
        st.markdown(f"""
        <div class="chart-container">
            <canvas id="forecastBarChart" width="400" height="200"></canvas>
        </div>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script>
            const ctx1 = document.getElementById('forecastBarChart').getContext('2d');
            new Chart(ctx1, {{
                type: 'bar',
                data: {{
                    labels: ['2025-01', '2025-02', '2025-03', '2025-04'],
                    datasets: [{{
                        label: 'Forecast',
                        data: {forecast_values},
                        backgroundColor: 'rgba(102, 126, 234, 0.8)',
                        borderColor: 'rgba(102, 126, 234, 1)',
                        borderWidth: 1
                    }}]
                }},
                options: {{
                    responsive: true,
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            max: {max(forecast_values) * 1.2},
                            ticks: {{
                                stepSize: {max(forecast_values) // 3}
                            }}
                        }}
                    }}
                }}
            }});
        </script>
        """, unsafe_allow_html=True)
    
    with col2:
        # Line chart untuk history vs forecast
        # Get real historical and forecast data if available
        history_data = [120000, 135000, 110000, 125000, 140000, 130000, 145000, 150000]  # Default values
        forecast_data = [1316571, 1358000, 1280000, 1420000]  # Default values
        
        if realtime_data:
            # Extract historical data from uploaded dataset
            uploaded_data = st.session_state.get('uploaded_data')
            if uploaded_data is not None:
                # Get historical data for the first part
                first_part = realtime_data[0] if realtime_data else {}
                part_no = first_part.get('part_number', '')
                if part_no:
                    part_data = uploaded_data[uploaded_data['PART_NO'] == part_no].sort_values('MONTH')
                    if len(part_data) >= 8:
                        history_data = part_data['ORIGINAL_SHIPPING_QTY'].tail(8).tolist()
            
            # Extract forecast data
            first_part = realtime_data[0] if realtime_data else {}
            forecast_data = [
                first_part.get('forecast_2025_01', 1316571),
                first_part.get('forecast_2025_02', 1358000),
                first_part.get('forecast_2025_03', 1280000),
                first_part.get('forecast_2025_04', 1420000)
            ]
        
        # Combine data for chart
        all_data = history_data + [None] * 4 + forecast_data
        all_labels = ['2024-05', '2024-06', '2024-07', '2024-08', '2024-09', '2024-10', '2024-11', '2024-12', '2025-01', '2025-02', '2025-03', '2025-04']
        
        st.markdown(f"""
        <div class="chart-container">
            <canvas id="historyForecastChart" width="400" height="200"></canvas>
        </div>
        <script>
            const ctx2 = document.getElementById('historyForecastChart').getContext('2d');
            new Chart(ctx2, {{
                type: 'line',
                data: {{
                    labels: {all_labels},
                    datasets: [{{
                        label: 'History',
                        data: {all_data},
                        borderColor: 'rgba(128, 128, 128, 1)',
                        backgroundColor: 'rgba(128, 128, 128, 0.1)',
                        fill: false,
                        tension: 0.1
                    }}, {{
                        label: 'Forecast',
                        data: {[None] * 8 + forecast_data},
                        borderColor: 'rgba(102, 126, 234, 1)',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        fill: false,
                        tension: 0.1
                    }}]
                }},
                options: {{
                    responsive: true,
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            max: {max(max(history_data), max(forecast_data)) * 1.2},
                            ticks: {{
                                stepSize: {max(max(history_data), max(forecast_data)) // 4}
                            }}
                        }}
                    }}
                }}
            }});
        </script>
        """, unsafe_allow_html=True)

def create_dashboard_training():
    """Create training results dashboard like in the image"""
    st.markdown("""
    <div class="dashboard-card">
        <div class="dashboard-title">Hasil Pelatihan Prediksi di 4 Bulan Sebelumnya</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Get real data if available
    forecast_data = st.session_state.get('forecast_results', {})
    backtest_data = forecast_data.get('data', {}).get('backtest_results', [])
    
    # Default values
    forecast_qty = "4,081,515"
    avg_error = "13.67%"
    
    # Update with real data if available
    if backtest_data:
        total_forecast = sum(item.get('forecast_value', 0) for item in backtest_data)
        avg_error_val = forecast_data.get('data', {}).get('average_error', 13.67)
        forecast_qty = f"{total_forecast:,.0f}"
        avg_error = f"{avg_error_val:.2f}%"
    
    # Metrics cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Forecast QTY</div>
            <div class="metric-value">{forecast_qty}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Average Error</div>
            <div class="metric-value">{avg_error}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts
        col1, col2 = st.columns(2)
        
        with col1:
        # Line chart untuk forecast vs actual
        # Get real backtest data if available
        forecast_data_points = [580000, 620000, 590000, 650000]  # Default values
        actual_data_points = [550000, 600000, 580000, 620000]  # Default values
        
        if backtest_data:
            # Extract forecast and actual values from real data
            forecast_data_points = []
            actual_data_points = []
            for item in backtest_data[-4:]:  # Get last 4 months
                forecast_data_points.append(item.get('forecast_value', 0))
                actual_data_points.append(item.get('actual_value', 0))
        
        st.markdown(f"""
        <div class="chart-container">
            <canvas id="forecastActualChart" width="400" height="200"></canvas>
        </div>
        <script>
            const ctx3 = document.getElementById('forecastActualChart').getContext('2d');
            new Chart(ctx3, {{
                type: 'line',
                data: {{
                    labels: ['2024-09', '2024-10', '2024-11', '2024-12'],
                    datasets: [{{
                        label: 'Forecast',
                        data: {forecast_data_points},
                        borderColor: 'rgba(102, 126, 234, 1)',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        fill: false,
                        tension: 0.1
                    }}, {{
                        label: 'Actual',
                        data: {actual_data_points},
                        borderColor: 'rgba(128, 128, 128, 1)',
                        backgroundColor: 'rgba(128, 128, 128, 0.1)',
                        fill: false,
                        tension: 0.1
                    }}]
                }},
                options: {{
                    responsive: true,
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            max: {max(max(forecast_data_points), max(actual_data_points)) * 1.2},
                            ticks: {{
                                stepSize: {max(max(forecast_data_points), max(actual_data_points)) // 6}
                            }}
                        }}
                    }}
                }}
            }});
        </script>
        """, unsafe_allow_html=True)
    
    with col2:
        # Bar chart untuk model performance
        # Get real model performance data if available
        model_performance = [2.8, 2.5, 1.8]  # Default values
        model_labels = ['WMA', 'RF', 'XGB']
        
        if backtest_data:
            # Extract model performance from real data
            model_scores = {}
            for item in backtest_data:
                best_model = item.get('best_model', 'WMA')
                if best_model not in model_scores:
                    model_scores[best_model] = 0
                model_scores[best_model] += 1
            
            # Convert to performance scores
            total_parts = len(backtest_data)
            model_performance = [
                model_scores.get('WMA', 0) / total_parts * 3,
                model_scores.get('RF', 0) / total_parts * 3,
                model_scores.get('XGB', 0) / total_parts * 3
            ]
        
        st.markdown(f"""
        <div class="chart-container">
            <canvas id="modelPerformanceChart" width="400" height="200"></canvas>
        </div>
        <script>
            const ctx4 = document.getElementById('modelPerformanceChart').getContext('2d');
            new Chart(ctx4, {{
                type: 'bar',
                data: {{
                    labels: {model_labels},
                    datasets: [{{
                        label: 'Performance Score',
                        data: {model_performance},
                        backgroundColor: 'rgba(102, 126, 234, 0.8)',
                        borderColor: 'rgba(102, 126, 234, 1)',
                        borderWidth: 1
                    }}]
                }},
                options: {{
                    responsive: true,
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            max: {max(model_performance) * 1.2},
                            ticks: {{
                                stepSize: {max(model_performance) // 6}
                            }}
                        }}
                    }}
                }}
            }});
        </script>
        """, unsafe_allow_html=True)

def main():
    """Main Streamlit application dengan layout seperti gambar"""
    
    # Welcome section
    st.markdown("""
    <div class="welcome-section">
        <div class="welcome-title">Welcome to Forecasting Parts Website</div>
        <div class="welcome-subtitle">
            Upload Dataset (Excel File) Anda dan Dapatkan Prediksi Permintaan Part Number Kamu di Bulan Mendatang
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Upload section
    create_upload_section()
    
    # Dashboard section
    st.markdown("""
    <div class="dashboard-container">
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        create_dashboard_prediction()
    
    with col2:
        create_dashboard_training()
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Show results if available
    if 'forecast_results' in st.session_state:
        st.markdown("---")
        st.header("📊 Forecast Results")
        display_forecast_results(st.session_state['forecast_results'])
    
    if 'categorization_results' in st.session_state:
    st.markdown("---")
        st.header("📋 Categorization Results")
        display_categorization_results(st.session_state['categorization_results'])

if __name__ == "__main__":
    # Initialize session state
    if 'session_id' not in st.session_state:
        st.session_state['session_id'] = f"session_{int(time.time())}"
    
    # Run main application
    main()
