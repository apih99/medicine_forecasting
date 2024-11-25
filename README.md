# Medicine Cost Forecasting System

A professional Streamlit application for analyzing and forecasting medicine costs in healthcare facilities.

## Features

- Interactive data analysis and visualization
- Multiple forecasting models (Random Forest, SVR, Linear Regression, ARIMA)
- Future cost predictions with confidence intervals
- Downloadable reports and forecasts
- Beautiful and intuitive user interface

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/medicine_forecasting.git
cd medicine_forecasting
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run Home.py
```

2. Open your web browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

3. Upload your medicine consumption data (CSV format) with the following required columns:
   - DocDate: Date of the transaction (format: DD/MM/YYYY)
   - DrugName: Name of the medicine
   - ItemValue: Cost of the medicine

4. Navigate through the different pages using the sidebar:
   - Home: Upload data and view basic statistics
   - Data Analysis: Explore historical patterns and trends
   - Model Training: Train and evaluate forecasting models
   - Forecasting: Generate and visualize future predictions
   - About: Learn more about the system

## Data Format

Your input CSV file should have the following structure:

```csv
DocDate,DrugName,ItemValue
01/01/2023,Drug A,100.50
01/01/2023,Drug B,75.25
02/01/2023,Drug A,95.75
...
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please contact support@example.com or open an issue on GitHub. 
