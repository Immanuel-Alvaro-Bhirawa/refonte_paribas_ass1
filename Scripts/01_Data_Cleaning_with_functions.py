def block_2():
    import pandas as pd
    Path to the Excel file
    NOTE: File_path should be the file path where you store the dataset
    file_path = "./CAC 40 Historical Data.xlsx"
    Read the Excel file
    df = pd.read_excel(file_path)
    Display the sampled DataFrame
    df

def block_4():
    df.dtypes

def block_6():
    Convert the 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'])
    Extracting Year, Month, and Day from the 'Date' column
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['unique_id'] = range(len(df), 0, -1)
    Adding 'Day name' where 0 is Sunday and 6 is Saturday
    df['Day name'] = df['Date'].dt.dayofweek.apply(lambda x: (x+2) % 7)
    Displaying the DataFrame to verify the new columns
    df

def block_8():
    df_cp = df['Price']
    Drop the "Closing Price" column
    df = df.drop('Price', axis=1)
    Create the "Date" column in "yyyy-mm-dd" format
    df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    Rejoin the "Closing Price" column (assuming you have it in another DataFrame)
    closing_prices_df = pd.DataFrame(df_cp)
    df = df.join(closing_prices_df)
    df

def block_10():
    Sort the DataFrame by "Date" column in descending order
    df = df.sort_values(by='Date', ascending=False)
    Print the sorted DataFrame
    df

def block_12():
    Assuming df is your DataFrame
    df['Date'] = pd.to_datetime(df['Date'])  # Ensure 'Date' is in datetime format
    df.set_index('Date', inplace=True)
    Create a continuous date range from min to max date in the dataset
    date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')

    df.reset_index(inplace=True)
    df

def block_14():
    import numpy as np
    from scipy.interpolate import Akima1DInterpolator
    Assuming 'df' is already defined and includes a 'Date' and 'Closing Price' column
    df.set_index('Date', inplace=True)
    Create a continuous date range from min to max date in the dataset
    date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    Reindex the dataframe with the full date range, filling non-existing dates with NaNs
    data_full = df.reindex(date_range)
    Prepare data for interpolation
    Dropping NaNs because Akima interpolation cannot handle NaNs directly
    x = np.arange(len(data_full))
    y = data_full['Price'].values
    mask = ~np.isnan(y)
    x, y = x[mask], y[mask]
    Create an Akima interpolator
    akima_interpolator = Akima1DInterpolator(x, y)
    Interpolate the results for the full range
    data_full['y'] = akima_interpolator(np.arange(len(data_full)))
    Generate a list of columns to forward fill, excluding 'Date' and 'Price'
    'Date' is not listed as it's the index, and 'Price' is explicitly handled above
    columns_to_ffill = [col for col in df.columns if col not in ['Price']]
    Forward fill the other columns where applicable
    data_full[columns_to_ffill] = data_full[columns_to_ffill].ffill()
    Save or display the result
    data_full.to_excel('Interpolated_Data.xlsx')
    print(data_full.head())


def block_15():
    data_full = data_full.drop('Price', axis = 1)
    data_full

def block_16():
    data_full.reset_index(inplace=True)
    data_full.rename(columns={'index': 'Date'}, inplace=True)
    Convert the 'Date' column to datetime type if it's not already
    data_full['Date'] = pd.to_datetime(data_full['Date'])
    data_full['Price'] = data_full['Price'].str.replace('.', '').str.replace(',', '.').astype(float)
    data_full

def block_18():
    def reset_unique_id(df):
    Calculating the total number of rows
        total_rows = df.shape[0]
    Creating a range from 1 to total_rows
        sequential_unique_id = range(1, total_rows + 1)
    Assigning the range to the 'unique_id' column
        df['unique_id'] = sequential_unique_id
        return df
    Apply the function to your DataFrame
    data_full = reset_unique_id(data_full)
    df = reset_unique_id(df)
    data_full

def block_20():
    # Reset the index without dropping it
    df = df.reset_index(drop=False)
    Rename the 'index' column to 'ds'
    df.rename(columns={'Date': 'ds'}, inplace=True)
    data_full.rename(columns={'Date': 'ds'}, inplace=True)
    df

def block_22():
    import pandas as pd
    import matplotlib.pyplot as plt
    Convert to datetime and rename the columns to comply with the library's expectations
    df.rename(columns={'Price': 'y'}, inplace=True)

    df

def block_23():
    df.dtypes

def block_24():
    data_full.dtypes

def block_25():
    data_full

def block_26():
    import pandas as pd
    Assuming 'data_full' is your DataFrame, if not, load your DataFrame here
    data_full = pd.read_csv('your_data_file.csv')  # Replace with your actual data loading code
    Define the output path
    output_path = r"C:\Users\Imman\Documents\Internship\Refonte_Paribas\Assignment_1\refonte_paribas_ass1\Data\Processed\CAC 40 Historical Data.xlsx"
    Export the DataFrame to the specified Excel file
    data_full.to_excel(output_path, index=False)

def block_27():
    print("Data Cleaning Finished!")

def block_28():
