import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Data Analysis", page_icon="📊", layout="wide")

st.markdown("# 📊 Data Analysis")
st.markdown("### Explore and analyze your medicine cost data")

if 'data' not in st.session_state:
    st.warning("Please upload your data first in the Home page!")
    st.stop()

df = st.session_state['data']

# Sidebar filters
st.sidebar.header("Filters")
selected_drugs = st.sidebar.multiselect(
    "Select Drugs",
    options=sorted(df['DrugName'].unique()),
    default=sorted(df['DrugName'].unique())[:3]
)

# Date range filter
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(df['DocDate'].min(), df['DocDate'].max()),
    min_value=df['DocDate'].min().date(),
    max_value=df['DocDate'].max().date()
)

# Filter data based on selection
filtered_df = df[
    (df['DrugName'].isin(selected_drugs)) &
    (df['DocDate'].dt.date >= date_range[0]) &
    (df['DocDate'].dt.date <= date_range[1])
]

# Create tabs for different analyses
tab1, tab2, tab3 = st.tabs(["Time Series Analysis", "Cost Distribution", "Summary Statistics"])

with tab1:
    st.markdown("### Cost Trends Over Time")
    
    # Daily cost trends
    daily_costs = filtered_df.groupby(['DocDate', 'DrugName'])['ItemValue'].sum().reset_index()
    
    fig = px.line(daily_costs, x='DocDate', y='ItemValue', color='DrugName',
                  title='Daily Cost Trends by Drug',
                  labels={'DocDate': 'Date', 'ItemValue': 'Cost', 'DrugName': 'Drug Name'})
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("### Cost Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Box plot
        fig_box = px.box(filtered_df, x='DrugName', y='ItemValue',
                        title='Cost Distribution by Drug',
                        labels={'DrugName': 'Drug Name', 'ItemValue': 'Cost'})
        fig_box.update_layout(height=500)
        st.plotly_chart(fig_box, use_container_width=True)
    
    with col2:
        # Violin plot
        fig_violin = px.violin(filtered_df, x='DrugName', y='ItemValue',
                             title='Cost Density Distribution',
                             labels={'DrugName': 'Drug Name', 'ItemValue': 'Cost'})
        fig_violin.update_layout(height=500)
        st.plotly_chart(fig_violin, use_container_width=True)

with tab3:
    st.markdown("### Summary Statistics")
    
    # Calculate summary statistics
    summary_stats = filtered_df.groupby('DrugName').agg({
        'ItemValue': ['count', 'mean', 'std', 'min', 'max', 'sum']
    }).round(2)
    
    summary_stats.columns = ['Count', 'Mean', 'Std Dev', 'Min', 'Max', 'Total Cost']
    summary_stats = summary_stats.reset_index()
    
    # Display summary statistics
    st.dataframe(summary_stats, use_container_width=True)
    
    # Create a pie chart of total costs
    fig_pie = px.pie(summary_stats, values='Total Cost', names='DrugName',
                     title='Proportion of Total Costs by Drug')
    st.plotly_chart(fig_pie, use_container_width=True)

# Download section
st.markdown("### 📥 Download Analysis Results")
csv = summary_stats.to_csv(index=False)
st.download_button(
    label="Download Summary Statistics as CSV",
    data=csv,
    file_name="drug_cost_analysis.csv",
    mime="text/csv",
) 