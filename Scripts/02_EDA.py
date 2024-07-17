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

