import warnings
import os
import pickle
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ValueWarning


# Suppress runtime and value warnings
os.environ['PYTHONWARNINGS'] = 'ignore::RuntimeWarning'
warnings.filterwarnings("ignore", category=ValueWarning)


def read_remote_data(url):
    """
    Read revenue and weather data from a remote excel file
    """
    # Load data

def preprocess_data(revenue_data, weather_data):
    """
    Function to preprocess revenue and weather data
    """
    # Drop the 'time' column from the weather_data

    # Convert categorical features 'wind' and 'condition' to numerical values

    # Reseample revenue_data

    # Aggregate weather_data on 'dt' to monthly level

    # Merge revenue_data and monthly_weather_data on the Date and dt columns

    # Drop the 'dt' column since it's a duplicate of 'Date'

    # Add the lagged revenue column to the merged_data

    # Replace zeros with NaNs

    # Drop column 'precipitation' with NaNs values

    # Drop rows with NaN values

    # Filter rows that have Revenue less than or equal to 10^9


def predict_next_month(model, preprocessed_data):
    """
    Predict the next month's revenue
    """
    # Split the data into train and test sets
    

    # Prepare the time series data for the ARIMA model
    
    
    # Get the forecast for the next month
    

    # Extract the last predicted value for the next month's revenue




def create_and_save_arima_model(ts_data, p=2, d=1, q=5):
    """
    Create and save the ARIMA model
    """
    # Create and fit the ARIMA model
    

    # Save the fitted ARIMA model
    

if __name__ == "__main__":
    url = "https://github.com/aurimas13/CodeAcademy-AI-Course/blob/main/Datasets/Revenue_Prognosis.xlsx?raw=True"   
    # Read the remote data using the read_remote_data function

    # Preprocess the revenue and weather data

    # Split the preprocessed data into training and testing sets
    
    # Create and save the ARIMA model

    # Load the ARIMA model
    with open('arima_results.pkl', 'rb') as f:
        model = pickle.load(f)
