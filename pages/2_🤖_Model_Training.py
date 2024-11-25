import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Model Training", page_icon="🤖", layout="wide")

st.markdown("# 🤖 Model Training")
st.markdown("### Train and evaluate forecasting models")

if 'data' not in st.session_state:
    st.warning("Please upload your data first in the Home page!")
    st.stop()

# Helper functions
def create_features(df):
    """Create time series features based on time series index."""
    features = pd.DataFrame(index=df.index)
    features['year'] = features.index.year
    features['month'] = features.index.month
    features['day'] = features.index.day
    features['day_of_week'] = features.index.dayofweek
    features['week_of_year'] = features.index.isocalendar().week
    features['quarter'] = features.index.quarter
    return features

def prepare_time_series_data(data, drug_name):
    """Prepare time series data for a specific drug"""
    drug_data = data[data['DrugName'] == drug_name].copy()
    drug_data = drug_data.set_index('DocDate')
    drug_data.index = pd.to_datetime(drug_data.index)
    drug_data = drug_data.sort_index()
    ts_data = drug_data['ItemValue'].resample('D').sum().fillna(0)
    ts_data = pd.DataFrame(ts_data)
    return ts_data

# Sidebar for model configuration
st.sidebar.header("Model Configuration")

# Select drug
df = st.session_state['data']
selected_drug = st.sidebar.selectbox(
    "Select Drug for Training",
    options=sorted(df['DrugName'].unique())
)

# Model selection
available_models = {
    'Random Forest': RandomForestRegressor,
    'SVR': SVR,
    'Linear Regression': LinearRegression,
    'ARIMA': ARIMA
}

selected_models = st.sidebar.multiselect(
    "Select Models to Train",
    options=list(available_models.keys()),
    default=['Random Forest', 'Linear Regression']
)

# Training parameters
train_size = st.sidebar.slider("Training Data Size (%)", 60, 90, 80)
rf_n_estimators = st.sidebar.number_input("Random Forest n_estimators", 50, 500, 100, 50)

# Train button
if st.sidebar.button("Train Models"):
    with st.spinner("Training models..."):
        # Prepare data
        ts_data = prepare_time_series_data(df, selected_drug)
        X, y = create_features(ts_data), ts_data['ItemValue']
        
        # Split data
        split_idx = int(len(X) * train_size / 100)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Initialize results storage
        results = []
        predictions = {}
        
        # Train and evaluate models
        for model_name in selected_models:
            if model_name == 'ARIMA':
                try:
                    model = ARIMA(y_train, order=(1,1,1))
                    model_fit = model.fit()
                    pred = model_fit.forecast(steps=len(y_test))
                except:
                    st.error(f"ARIMA model failed to converge for {selected_drug}")
                    continue
            else:
                if model_name == 'Random Forest':
                    model = available_models[model_name](n_estimators=rf_n_estimators, random_state=42)
                else:
                    model = available_models[model_name]()
                
                model.fit(X_train_scaled, y_train)
                pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, pred)
            r2 = r2_score(y_test, pred)
            
            results.append({
                'Model': model_name,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            })
            predictions[model_name] = pred
        
        # Display results
        st.markdown("### Model Performance Metrics")
        results_df = pd.DataFrame(results)
        st.dataframe(results_df)
        
        # Visualize predictions
        st.markdown("### Prediction Visualization")
        fig = go.Figure()
        
        # Add actual values
        fig.add_trace(go.Scatter(
            x=X_test.index,
            y=y_test,
            name='Actual',
            line=dict(color='black', width=2)
        ))
        
        # Add predictions for each model
        colors = px.colors.qualitative.Set3
        for i, (model_name, pred) in enumerate(predictions.items()):
            fig.add_trace(go.Scatter(
                x=X_test.index,
                y=pred,
                name=f'{model_name} Prediction',
                line=dict(color=colors[i], width=2, dash='dash')
            ))
        
        fig.update_layout(
            title=f'Cost Forecasting Results for {selected_drug}',
            xaxis_title='Date',
            yaxis_title='Cost',
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Save results to session state for use in other pages
        st.session_state['model_results'] = {
            'drug': selected_drug,
            'metrics': results_df,
            'predictions': predictions,
            'test_data': {
                'X': X_test,
                'y': y_test
            }
        }
        
        st.success("Model training completed successfully!")

# Display helpful information
with st.expander("ℹ️ How to use this page"):
    st.markdown("""
    1. Select a drug from the sidebar dropdown
    2. Choose which models you want to train
    3. Adjust the training parameters if needed
    4. Click 'Train Models' to start the training process
    5. Review the results and visualizations
    
    The metrics shown are:
    - **RMSE**: Root Mean Square Error (lower is better)
    - **MAE**: Mean Absolute Error (lower is better)
    - **R²**: R-squared score (higher is better, max 1.0)
    """) 