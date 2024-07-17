import pandas as pd

# Path to the Excel file
# NOTE: File_path should be the file path where you store the dataset
file_path = "./CAC 40 Historical Data.xlsx"

# Read the Excel file
df = pd.read_excel(file_path)

# Display the sampled DataFrame
df

df.dtypes

# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Extracting Year, Month, and Day from the 'Date' column
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['unique_id'] = range(len(df), 0, -1)

# Adding 'Day name' where 0 is Sunday and 6 is Saturday
df['Day name'] = df['Date'].dt.dayofweek.apply(lambda x: (x+2) % 7)

# Displaying the DataFrame to verify the new columns
df

df_cp = df['Price']
# Drop the "Closing Price" column
df = df.drop('Price', axis=1)

# Create the "Date" column in "yyyy-mm-dd" format
# df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])

# Rejoin the "Closing Price" column (assuming you have it in another DataFrame)
closing_prices_df = pd.DataFrame(df_cp)
df = df.join(closing_prices_df)
df

# Sort the DataFrame by "Date" column in descending order
df = df.sort_values(by='Date', ascending=False)

# Print the sorted DataFrame
df

# Assuming df is your DataFrame
df['Date'] = pd.to_datetime(df['Date'])  # Ensure 'Date' is in datetime format
df.set_index('Date', inplace=True)

# Create a continuous date range from min to max date in the dataset
date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')

df.reset_index(inplace=True)
df

import numpy as np
from scipy.interpolate import Akima1DInterpolator

# Assuming 'df' is already defined and includes a 'Date' and 'Closing Price' column
df.set_index('Date', inplace=True)

# Create a continuous date range from min to max date in the dataset
date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')

# Reindex the dataframe with the full date range, filling non-existing dates with NaNs
data_full = df.reindex(date_range)

# Prepare data for interpolation
# Dropping NaNs because Akima interpolation cannot handle NaNs directly
x = np.arange(len(data_full))
y = data_full['Price'].values
mask = ~np.isnan(y)
x, y = x[mask], y[mask]

# Create an Akima interpolator
akima_interpolator = Akima1DInterpolator(x, y)

# Interpolate the results for the full range
data_full['y'] = akima_interpolator(np.arange(len(data_full)))

# Generate a list of columns to forward fill, excluding 'Date' and 'Price'
# 'Date' is not listed as it's the index, and 'Price' is explicitly handled above
columns_to_ffill = [col for col in df.columns if col not in ['Price']]

# Forward fill the other columns where applicable
data_full[columns_to_ffill] = data_full[columns_to_ffill].ffill()

# Save or display the result
# data_full.to_excel('Interpolated_Data.xlsx')
print(data_full.head())


data_full = data_full.drop('Price', axis = 1)
data_full

data_full.reset_index(inplace=True)
data_full.rename(columns={'index': 'Date'}, inplace=True)

# Convert the 'Date' column to datetime type if it's not already
data_full['Date'] = pd.to_datetime(data_full['Date'])
# data_full['Price'] = data_full['Price'].str.replace('.', '').str.replace(',', '.').astype(float)
data_full

def reset_unique_id(df):
    # Calculating the total number of rows
    total_rows = df.shape[0]

    # Creating a range from 1 to total_rows
    sequential_unique_id = range(1, total_rows + 1)

    # Assigning the range to the 'unique_id' column
    df['unique_id'] = sequential_unique_id
    return df

# Apply the function to your DataFrame
data_full = reset_unique_id(data_full)
df = reset_unique_id(df)
data_full

# # Reset the index without dropping it
df = df.reset_index(drop=False)

# Rename the 'index' column to 'ds'
df.rename(columns={'Date': 'ds'}, inplace=True)
data_full.rename(columns={'Date': 'ds'}, inplace=True)
df

import pandas as pd
import matplotlib.pyplot as plt

# Convert to datetime and rename the columns to comply with the library's expectations
df.rename(columns={'Price': 'y'}, inplace=True)

df

df.dtypes

data_full.dtypes

data_full

plt.figure(figsize=(20, 5))
plt.plot(data_full.ds, data_full['y'])  # Plotting the data
plt.title('FCHI CAC 40 Stock Price over time')
plt.ylabel('Price')
plt.xlabel('Date')
plt.grid(True)
plt.show()

import plotly.express as px
import plotly.graph_objects as go
# A pro technique would be to use plotly for interactive visual and time selectors

fig = px.line(data_full, x='ds', y='y', title='Time Series with Selectors')

fig.update_xaxes(
    rangeslider_visible=False,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)
fig.show()

from statsmodels.tsa.stattools import adfuller
test_result=adfuller(data_full['y'])

#Ho: Data is non stationary
#H1: Data is stationary

def adfuller_test(price):
    result=adfuller(price)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")

adfuller_test(data_full['y'])

from statsmodels.tsa.seasonal import seasonal_decompose
plt.rcParams.update({'figure.figsize': (10,10)})
y = data_full['y'].to_frame()


# Multiplicative Decomposition 
result_mul = seasonal_decompose(y, model='multiplicative',period = 52)

# Additive Decomposition
result_add = seasonal_decompose(y, model='additive',period = 52)

# Plot
plt.rcParams.update({'figure.figsize': (10,10)})
result_mul.plot().suptitle('Multiplicative Decompose', fontsize=22)
result_add.plot().suptitle('Additive Decompose', fontsize=22)
plt.show()

data_full['y_diff']=data_full['y']-data_full['y'].shift(1)

import plotly.graph_objects as go
fig = go.Figure([go.Scatter(x=data_full.index,y=data_full.y)])
fig.update_layout(
    autosize=False,
    width=1000,
    height=500,
    template='simple_white',
    title='Closing Price Over Time ')
fig.show()

data_full = data_full.drop('y_diff', axis=1)
data_full

from sklearn.model_selection import train_test_split as TTS
df_dataCol = data_full.drop('y', axis = 1)

X = data_full[df_dataCol.columns]
Y = data_full['y']
# Y = df["Closing Price"] / df["Closing Price"].max()

X_train, X_test, Y_train, Y_test = TTS(X, Y, 
                                       test_size = 0.05, 
                                       random_state = 0,
                                       shuffle=False)

df_train = X_train.join(Y_train)
df_train

df_test = X_test.join(Y_test)
df_test

from statsforecast.adapters.prophet import AutoARIMAProphet
from tqdm import tqdm
import time

start = time.time()
# Initialize the AutoARIMAProphet model configurations
model_config = {
    "growth": "logistic",
    "yearly_seasonality": True,
    # "weekly_seasonality": True,
    # "daily_seasonality" : False,
    # "holidays": holidays_df
    "seasonality_mode": "multiplicative",
    "seasonality_prior_scale": 10,
    "holidays_prior_scale": 10,
    "changepoint_prior_scale": 0.05,
    "interval_width": 0.75,
    "uncertainty_samples": 1000
}

cap = 10
floor = 5.5

# Instantiate models
aap = AutoARIMAProphet(**model_config)
aapM2 = AutoARIMAProphet(**model_config)

df_train['cap'] = cap
df_train['floor'] = floor

# Fit the first model
with tqdm(total=1, desc="Fitting First Model") as pbar:
    aap1 = aap.fit(df_train, disable_seasonal_features=False)
    pbar.update(1)
    
# aap = aap.fit(df_train)
print("Train:", time.time() - start)

combined_df = pd.concat([df_train, df_test])

df_train['cap'] = cap
df_train['floor'] = floor

df_test['cap'] = cap
df_test['floor'] = floor

# with tqdm(total=100, desc="Making Predictions") as pbar:
#     aap_pred_test = aap1.predict(df_test)
#     aap_pred_train = aap1.predict(df_train)
#     pbar.update(100)

combined_df['cap'] = cap
combined_df['floor'] = floor

# Fit the second model
with tqdm(total=1, desc="Fitting Second Model") as pbar:
    aap2 = aapM2.fit(combined_df, disable_seasonal_features=False)
    pbar.update(1)


aap_pred_forecast = aap2.make_future_dataframe(periods = 1826, freq = 'D', include_history = True)
aap_pred_forecast['cap'] = cap
aap_pred_forecast['floor'] = floor

# Making predictions
with tqdm(total=2, desc="Making Predictions") as pbar:
    aap_pred_test = aap1.predict(df_test)
    aap_pred_train = aap1.predict(df_train)
    aap_pred_forecast = aap2.predict(aap_pred_forecast)
    pbar.update(2)

print("Pred:", time.time() - start)

aap_pred_forecast

def MAPE(actual, prediction):
    """
    Calculate the Mean Absolute Percentage Error (MAPE) between two lists.
    
    MAPE is a measure of prediction accuracy of a forecasting method in statistics,
    specifically trending markets. It expresses accuracy as a percentage, and is defined
    by the formula: MAPE = (1/n) * Σ(|actual - prediction| / |actual|) * 100
    
    Parameters:
    - actual (list or array-like): The actual data points. Must be 1-dimensional and 
      the same length as 'prediction'.
    - prediction (list or array-like): The predicted data points, which correspond to 
      'actual'. Must be 1-dimensional and the same length as 'actual'.

    Returns:
    - float: The mean absolute percentage error (MAPE) as a percentage, rounded to two decimal places.
    """
    actual, prediction = np.array(actual), np.array(prediction)
    if np.any(actual == 0):
        raise ValueError("MAPE is undefined for zero actual values because of division by zero")
    mape = np.mean(np.abs((actual - prediction) / actual)) * 100
    return round(mape, 2)

