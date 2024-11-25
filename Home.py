import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(
    page_title="Medicine Cost Forecasting",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 0rem;
    }
    .title-text {
        font-size: 40px;
        font-weight: bold;
        color: #800020;
        text-align: center;
        margin-bottom: 30px;
    }
    .subtitle-text {
        font-size: 24px;
        color: #800020;
        text-align: center;
        margin-bottom: 20px;
    }
    .sample-data {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 5px;
        margin: 20px 0;
        border: 1px solid #ddd;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: #ffffff;
        color: #1a1a1a;
        padding: 20px;
        border-radius: 5px;
        margin: 15px 0;
        border-left: 5px solid #2196F3;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stDataFrame {
        background-color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

# Title Section
st.markdown('<p class="title-text">Medicine Cost Forecasting System</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Advanced Analytics for Healthcare Cost Management</p>', unsafe_allow_html=True)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ### 🎯 Key Features
    - **Real-time Cost Analysis**: Monitor medicine costs with interactive dashboards
    - **Predictive Analytics**: Forecast future medicine costs using advanced ML models
    - **Multi-model Comparison**: Compare different forecasting models for better accuracy
    - **Interactive Visualizations**: Explore data through dynamic charts and graphs
    """)

with col2:
    st.markdown("""
    ### 🔍 Quick Navigation
    - **Data Analysis**: Explore historical cost data
    - **Model Training**: Train and evaluate forecasting models
    - **Forecasting**: View cost predictions
    - **About**: Learn more about the system
    """)

# Required Data Format Section
st.markdown("### 📋 Required Data Format")
st.markdown("""
<div class="info-box">
<strong>Your CSV file must contain the following columns:</strong><br><br>

1. <strong>DocDate</strong>: Date of the transaction (format: DD/MM/YYYY)<br>
2. <strong>DrugName</strong>: Name of the medicine<br>
3. <strong>ItemValue</strong>: Cost of the medicine (numeric value)
</div>
""", unsafe_allow_html=True)

# Sample Data Display
st.markdown("### 📝 Sample Data Format")
sample_data = pd.DataFrame({
    'DocDate': ['01/01/2023', '01/01/2023', '02/01/2023', '02/01/2023', '03/01/2023'],
    'DrugName': ['Paracetamol', 'Amoxicillin', 'Paracetamol', 'Ibuprofen', 'Amoxicillin'],
    'ItemValue': [100.50, 250.75, 98.25, 150.00, 245.50]
})

# Display sample data in a styled container
st.markdown('<div class="sample-data">', unsafe_allow_html=True)
st.markdown("#### Example CSV Content:")
st.dataframe(
    sample_data,
    use_container_width=True,
    hide_index=True
)

# Add download sample data button
csv = sample_data.to_csv(index=False)
st.download_button(
    label="📥 Download Sample CSV Template",
    data=csv,
    file_name="sample_medicine_data.csv",
    mime="text/csv",
)
st.markdown('</div>', unsafe_allow_html=True)

# Common Issues and Tips
st.markdown("### ℹ️ Tips for Data Preparation")
st.markdown("""
<div class="info-box">
<strong>Important considerations for your data:</strong><br><br>

1. <strong>Date Format</strong>: Ensure dates are in DD/MM/YYYY format<br>
2. <strong>Missing Values</strong>: Remove or handle any missing values in your data<br>
3. <strong>Consistency</strong>: Maintain consistent drug names throughout the dataset<br>
4. <strong>Values</strong>: Ensure ItemValue contains only numeric values<br>
5. <strong>Headers</strong>: Column names should match exactly as shown above
</div>
""", unsafe_allow_html=True)

# Data Upload Section
st.markdown("### 📤 Upload Your Data")
uploaded_file = st.file_uploader("Upload your medicine consumption data (CSV format)", type=['csv'])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df['DocDate'] = pd.to_datetime(df['DocDate'], format='%d/%m/%Y')
        
        # Display sample data
        st.markdown("### 📊 Data Preview")
        st.dataframe(df.head())
        
        # Show basic statistics
        st.markdown("### 📈 Basic Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Unique Drugs", df['DrugName'].nunique())
        with col3:
            st.metric("Date Range", f"{df['DocDate'].min().strftime('%Y-%m-%d')} to {df['DocDate'].max().strftime('%Y-%m-%d')}")
        
        # Save the dataframe to session state for other pages to use
        st.session_state['data'] = df
        st.success("Data loaded successfully! Navigate to other pages to start analysis.")
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Developed with ❤️ for Healthcare Analytics</p>
</div>
""", unsafe_allow_html=True) 