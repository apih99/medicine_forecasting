import streamlit as st

st.set_page_config(page_title="About", page_icon="ℹ️", layout="wide")

st.markdown("# ℹ️ About")
st.markdown("### Medicine Cost Forecasting System")

# Application Overview
st.markdown("""
## Overview

The Medicine Cost Forecasting System is a sophisticated analytics tool designed to help healthcare facilities better manage and predict their medicine costs. This application combines advanced time series analysis with machine learning techniques to provide accurate cost forecasts for various medicines.

### Key Features

1. **Data Analysis**
   - Interactive visualization of historical cost trends
   - Statistical analysis of cost distributions
   - Comprehensive summary statistics

2. **Model Training**
   - Multiple forecasting models:
     - Random Forest Regression
     - Support Vector Regression (SVR)
     - Linear Regression
     - ARIMA (Autoregressive Integrated Moving Average)
   - Model performance metrics and comparisons
   - Interactive parameter tuning

3. **Forecasting**
   - Future cost predictions with confidence intervals
   - Customizable forecast horizon
   - Downloadable forecast results
   - Visual representation of predictions

### How It Works

1. **Data Upload**
   - Upload your medicine consumption data in CSV format
   - The system automatically processes and validates the data
   - Supports multiple medicine types and time periods

2. **Analysis**
   - The system analyzes historical patterns and trends
   - Generates comprehensive visualizations
   - Provides statistical insights

3. **Model Training**
   - Select from multiple forecasting models
   - Customize training parameters
   - Evaluate model performance

4. **Forecasting**
   - Generate future cost predictions
   - View confidence intervals
   - Download detailed forecast reports

### Technical Details

The application uses several advanced technologies and libraries:
- **Streamlit**: For the interactive web interface
- **Pandas & NumPy**: For data manipulation and numerical computations
- **Scikit-learn**: For machine learning models
- **Plotly**: For interactive visualizations
- **Statsmodels**: For statistical modeling

### Best Practices for Use

1. **Data Preparation**
   - Ensure your data is clean and properly formatted
   - Include all relevant cost information
   - Maintain consistent date formats

2. **Model Selection**
   - Compare multiple models for best results
   - Consider the trade-off between accuracy and complexity
   - Regularly retrain models with new data

3. **Interpretation**
   - Consider confidence intervals when making decisions
   - Account for seasonal patterns and trends
   - Use multiple metrics to evaluate performance

### Support and Contact

For questions, support, or feature requests, please contact:
- Email: hafizcr716@gmail.com
- GitHub: [Project Repository](https://github.com/apih99/medicine_forecasting)

### Version Information
- Current Version: 1.0.0
- Last Updated: 2024
""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Developed with ❤️ for Healthcare Analytics</p>
    <p>© 2024 Medicine Cost Forecasting System</p>
</div>
""", unsafe_allow_html=True) 
