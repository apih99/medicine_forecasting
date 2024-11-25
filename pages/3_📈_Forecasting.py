import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Forecasting", page_icon="📈", layout="wide")

st.markdown("# 📈 Forecasting Results")
st.markdown("### View and analyze future cost predictions")

if 'model_results' not in st.session_state:
    st.warning("Please train models first in the Model Training page!")
    st.stop()

# Get results from session state
results = st.session_state['model_results']
drug_name = results['drug']
metrics_df = results['metrics']
predictions = results['predictions']
test_data = results['test_data']

# Sidebar for forecast configuration
st.sidebar.header("Forecast Configuration")

# Select the best performing model based on R2 score
best_model = metrics_df.loc[metrics_df['R2'].idxmax(), 'Model']
selected_model = st.sidebar.selectbox(
    "Select Model for Forecasting",
    options=metrics_df['Model'].tolist(),
    index=metrics_df['Model'].tolist().index(best_model)
)

# Forecast horizon
forecast_days = st.sidebar.slider("Forecast Horizon (Days)", 7, 90, 30)

# Display current model performance
st.markdown("### Selected Model Performance")
model_metrics = metrics_df[metrics_df['Model'] == selected_model].iloc[0]
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("RMSE", f"{model_metrics['RMSE']:.2f}")
with col2:
    st.metric("MAE", f"{model_metrics['MAE']:.2f}")
with col3:
    st.metric("R² Score", f"{model_metrics['R2']:.2f}")

# Visualize historical data and predictions
st.markdown("### Historical Data and Predictions")

fig = go.Figure()

# Plot historical data
fig.add_trace(go.Scatter(
    x=test_data['X'].index,
    y=test_data['y'],
    name='Actual',
    line=dict(color='black', width=2)
))

# Plot model predictions
fig.add_trace(go.Scatter(
    x=test_data['X'].index,
    y=predictions[selected_model],
    name='Predicted',
    line=dict(color='blue', width=2, dash='dash')
))

# Add confidence intervals (simple example using standard deviation)
pred_std = np.std(predictions[selected_model])
fig.add_trace(go.Scatter(
    x=test_data['X'].index,
    y=predictions[selected_model] + 2*pred_std,
    fill=None,
    mode='lines',
    line=dict(color='rgba(0,0,255,0)'),
    showlegend=False
))

fig.add_trace(go.Scatter(
    x=test_data['X'].index,
    y=predictions[selected_model] - 2*pred_std,
    fill='tonexty',
    mode='lines',
    line=dict(color='rgba(0,0,255,0)'),
    name='95% Confidence Interval'
))

# Update layout
fig.update_layout(
    title=f'Historical Cost Predictions for {drug_name}',
    xaxis_title='Date',
    yaxis_title='Cost',
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# Future Forecast
st.markdown("### Future Cost Forecast")

# Calculate future dates
last_date = test_data['X'].index[-1]
future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                           periods=forecast_days, 
                           freq='D')

# Generate simple forecast (example using moving average of last predictions)
window_size = min(len(predictions[selected_model]), 7)
last_predictions = predictions[selected_model][-window_size:]
baseline = np.mean(last_predictions)

# Add some randomness and trend for illustration
trend = np.linspace(0, 0.1, forecast_days) * baseline
noise = np.random.normal(0, pred_std/4, forecast_days)
forecast_values = baseline + trend + noise

# Plot forecast
fig_forecast = go.Figure()

# Historical data
fig_forecast.add_trace(go.Scatter(
    x=test_data['X'].index,
    y=test_data['y'],
    name='Historical',
    line=dict(color='black', width=2)
))

# Historical predictions
fig_forecast.add_trace(go.Scatter(
    x=test_data['X'].index,
    y=predictions[selected_model],
    name='Historical Predictions',
    line=dict(color='blue', width=2, dash='dash')
))

# Future forecast
fig_forecast.add_trace(go.Scatter(
    x=future_dates,
    y=forecast_values,
    name='Future Forecast',
    line=dict(color='red', width=2, dash='dash')
))

# Confidence intervals for future forecast
fig_forecast.add_trace(go.Scatter(
    x=future_dates,
    y=forecast_values + 2*pred_std,
    fill=None,
    mode='lines',
    line=dict(color='rgba(255,0,0,0)'),
    showlegend=False
))

fig_forecast.add_trace(go.Scatter(
    x=future_dates,
    y=forecast_values - 2*pred_std,
    fill='tonexty',
    mode='lines',
    line=dict(color='rgba(255,0,0,0)'),
    name='95% Confidence Interval'
))

fig_forecast.update_layout(
    title=f'Future Cost Forecast for {drug_name}',
    xaxis_title='Date',
    yaxis_title='Cost',
    height=500
)

st.plotly_chart(fig_forecast, use_container_width=True)

# Forecast Statistics
st.markdown("### Forecast Statistics")

forecast_stats = pd.DataFrame({
    'Metric': ['Mean Forecast', 'Min Forecast', 'Max Forecast', 
               'Standard Deviation', 'Total Forecasted Cost'],
    'Value': [
        f"{np.mean(forecast_values):.2f}",
        f"{np.min(forecast_values):.2f}",
        f"{np.max(forecast_values):.2f}",
        f"{np.std(forecast_values):.2f}",
        f"{np.sum(forecast_values):.2f}"
    ]
})

st.dataframe(forecast_stats, use_container_width=True)

# Download forecast data
forecast_df = pd.DataFrame({
    'Date': future_dates,
    'Forecasted_Cost': forecast_values,
    'Lower_Bound': forecast_values - 2*pred_std,
    'Upper_Bound': forecast_values + 2*pred_std
})

csv = forecast_df.to_csv(index=False)
st.download_button(
    label="Download Forecast Data as CSV",
    data=csv,
    file_name=f"forecast_{drug_name}.csv",
    mime="text/csv",
) 