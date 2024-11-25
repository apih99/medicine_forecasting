# Import required libraries
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
import warnings

# Configuration
warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_theme(style='whitegrid')

# Data Loading and Preprocessing Functions
def load_data(file_path):
    """Load and preprocess the initial dataset"""
    df = pd.read_csv(file_path)
    df['DocDate'] = pd.to_datetime(df['DocDate'], format='%d/%m/%Y')
    df = df.sort_values('DocDate')
    return df

def create_daily_demand(df):
    """Create daily aggregated data for each drug"""
    daily_demand = df.groupby(['DocDate', 'DrugName'])['IssueQty'].sum().reset_index()
    return daily_demand

def prepare_time_series_data(data, drug_name):
    """Prepare time series data for a specific drug"""
    drug_data = data[data['DrugName'] == drug_name].copy()
    drug_data = drug_data.set_index('DocDate')
    drug_data = drug_data.resample('D')['IssueQty'].sum().fillna(0)
    return drug_data

# Feature Engineering Functions
def create_features(data, lookback=7):
    """Create features for ML models using a sliding window approach"""
    X = []
    y = []
    values = data.values
    for i in range(len(values) - lookback):
        X.append(values[i:(i + lookback)])
        y.append(values[i + lookback])
    return np.array(X), np.array(y)

# Model Training and Evaluation Functions
def evaluate_model(y_true, y_pred, model_name):
    """Calculate and return model performance metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'Model': model_name,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

def train_evaluate_models(drug_name, data):
    """Train and evaluate all models for a specific drug"""
    print(f"\nAnalyzing demand for {drug_name}")
    
    # Prepare data
    ts_data = prepare_time_series_data(data, drug_name)
    
    # Split data into train and test sets
    train_size = int(len(ts_data) * 0.8)
    train_data = ts_data[:train_size]
    test_data = ts_data[train_size:]
    
    # Create features for ML models
    X_train_full, y_train_full = create_features(train_data)
    X_test_full, y_test_full = create_features(test_data)
    
    # Split training data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )
    
    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test_full.reshape(-1, X_test_full.shape[-1])).reshape(X_test_full.shape)
    
    results = []
    predictions = {}
    
    # Train and evaluate models
    # 1. ARIMA
    try:
        model_arima = ARIMA(train_data, order=(1,1,1))
        model_arima_fit = model_arima.fit()
        arima_pred = model_arima_fit.forecast(steps=len(test_data))
        results.append(evaluate_model(test_data.values, arima_pred, 'ARIMA'))
        predictions['ARIMA'] = arima_pred
    except:
        print("ARIMA model failed to converge")
    
    # 2. Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    results.append(evaluate_model(y_test_full, rf_pred, 'Random Forest'))
    predictions['Random Forest'] = rf_pred
    
    # 3. SVR
    svr_model = SVR(kernel='rbf')
    svr_model.fit(X_train_scaled, y_train)
    svr_pred = svr_model.predict(X_test_scaled)
    results.append(evaluate_model(y_test_full, svr_pred, 'SVR'))
    predictions['SVR'] = svr_pred
    
    # 4. Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    results.append(evaluate_model(y_test_full, lr_pred, 'Linear Regression'))
    predictions['Linear Regression'] = lr_pred
    
    # Visualize results
    plot_predictions(test_data, y_test_full, predictions, drug_name)
    
    return pd.DataFrame(results)

# Visualization Functions
def plot_predictions(test_data, y_test_full, predictions, drug_name):
    """Plot actual vs predicted values for all models"""
    plt.figure(figsize=(15, 6))
    
    actual_dates = test_data.index[7:]  # Adjust for lookback window
    plt.plot(actual_dates, y_test_full, label='Actual', color='black')
    
    for model_name, pred in predictions.items():
        if model_name == 'ARIMA':
            plt.plot(test_data.index, pred, label=model_name, alpha=0.7)
        else:
            plt.plot(actual_dates, pred, label=model_name, alpha=0.7)
    
    plt.title(f'Demand Forecasting Comparison for {drug_name}')
    plt.xlabel('Date')
    plt.ylabel('Demand Quantity')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_model_comparison(final_results):
    """Plot overall model performance comparison"""
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=final_results, x='Model', y='R2')
    plt.title('Overall Model Performance Comparison (R² Score)')
    plt.xlabel('Model')
    plt.ylabel('R² Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_drug_specific_performance(final_results):
    """Create a detailed visualization of R² scores for each drug and model"""
    # Prepare the data
    plt.figure(figsize=(15, 8))
    
    # Create grouped bar plot
    drugs = final_results['Drug'].unique()
    models = final_results['Model'].unique()
    x = np.arange(len(drugs))
    width = 0.2  # Width of the bars
    
    # Plot bars for each model
    for i, model in enumerate(models):
        model_data = final_results[final_results['Model'] == model]
        offset = (i - len(models)/2 + 0.5) * width
        plt.bar(x + offset, 
                model_data['R2'], 
                width, 
                label=model,
                alpha=0.8)
    
    # Customize the plot
    plt.xlabel('Drugs')
    plt.ylabel('R² Score')
    plt.title('Model Performance (R² Score) by Drug')
    plt.xticks(x, drugs, rotation=45, ha='right')
    plt.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add grid for better readability
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.show()

def format_results_table(final_results):
    """Format the results into a styled table"""
    # Round numeric columns to 4 decimal places
    formatted_results = final_results.copy()
    formatted_results['RMSE'] = formatted_results['RMSE'].round(4)
    formatted_results['MAE'] = formatted_results['MAE'].round(4)
    formatted_results['R2'] = formatted_results['R2'].round(4)
    
    # Create a styled table
    styled_table = formatted_results.style\
        .set_properties(**{
            'background-color': '#1e1e1e',
            'color': 'white',
            'border-color': '#444444',
            'padding': '10px'
        })\
        .set_table_styles([
            {'selector': 'th',
             'props': [('background-color', '#2d2d2d'),
                      ('color', 'white'),
                      ('font-weight', 'bold'),
                      ('padding', '10px'),
                      ('text-align', 'center')]},
            {'selector': 'td',
             'props': [('text-align', 'right'),
                      ('padding', '8px')]}
        ])\
        .format({
            'RMSE': '{:,.4f}',
            'MAE': '{:,.4f}',
            'R2': '{:,.4f}'
        })\
        .hide_index()
    
    return styled_table

# Main execution
def main():
    # Load and preprocess data
    df = load_data('Consumption Report by Issuing Store.csv')
    daily_demand = create_daily_demand(df)
    unique_drugs = daily_demand['DrugName'].unique()
    
    # Display initial information
    print("Dataset Overview:")
    print(daily_demand.head())
    print(f"\nNumber of unique drugs: {len(unique_drugs)}")
    print("\nUnique drugs:")
    print(unique_drugs)
    
    # Analyze each drug
    all_results = []
    for drug in unique_drugs:
        print(f"\nAnalyzing {drug}...")
        results = train_evaluate_models(drug, daily_demand)
        results['Drug'] = drug
        all_results.append(results)
    
    # Combine results
    final_results = pd.concat(all_results)
    
    # Display formatted results
    print("\nFinal Model Comparison:")
    display_df = final_results[['Drug', 'Model', 'RMSE', 'MAE', 'R2']].copy()
    
    # Create a more organized display with grouping by drug
    for drug in unique_drugs:
        print(f"\n{'='*80}")
        print(f"Results for {drug:}")
        print('='*80)
        drug_results = display_df[display_df['Drug'] == drug].copy()
        drug_results = drug_results.drop('Drug', axis=1)
        print(drug_results.to_string(index=False, float_format=lambda x: '{:.4f}'.format(x)))
    
    # Plot model comparisons
    plot_model_comparison(final_results)
    plot_drug_specific_performance(final_results)

if __name__ == "__main__":
    main()