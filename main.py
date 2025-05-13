import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Load Financial Dataset
file_path = r"C:\Users\Gouri\OneDrive\Documents\Projects_based_on_dataset\Financial Performance\financial_performance_data.csv"
df = pd.read_csv(file_path)

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Forecast Revenue & Expenses for each company
forecast_results = []
for company in df['Company'].unique():
    for metric in ['Revenue', 'Expenses']:
        company_data = df[df['Company'] == company][['Date', metric]].set_index('Date')

        # Check if enough data exists
        if len(company_data) > 3:
            # Apply Exponential Smoothing
            model = ExponentialSmoothing(company_data, trend="add", seasonal="add", seasonal_periods=12)
            fit = model.fit()
            future_dates = pd.date_range(start=company_data.index[-1], periods=13, freq='M')[1:]
            forecast = fit.forecast(len(future_dates))

            forecast_df = pd.DataFrame({'ds': future_dates, 'Company': company, 'Metric': metric, 'yhat': forecast.values})
            forecast_results.append(forecast_df)

# Save Forecasted Data
forecast_df = pd.concat(forecast_results)
forecast_df.to_csv("financial_forecast.csv", index=False)

# Risk Segmentation based on Profitability
risk_segments = []
for company in df['Company'].unique():
    company_data = df[df['Company'] == company]
    avg_profit = company_data['Profit'].mean()
    avg_debt = company_data['Debt'].mean()

    risk_category = "Low Risk" if avg_profit > 50000 else "High Risk" if avg_debt > 20000 else "Medium Risk"
    risk_segments.append([company, avg_profit, avg_debt, risk_category])

risk_df = pd.DataFrame(risk_segments, columns=['Company', 'Avg_Profit', 'Avg_Debt', 'Risk_Category'])
risk_df.to_csv("financial_risk_analysis.csv", index=False)

print("Financial Forecast & Risk Analysis Completed Successfully!")
