import pandas as pd
import numpy as np
import gzip
import os
import json
import ast
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pyspark.sql import SparkSession
from sklearn.preprocessing import KBinsDiscretizer


def load_csv(path):
    """
    Loads a CSV file and returns a pandas DataFrame.
    """
    dataset = pd.read_csv(path)
    df = pd.DataFrame(dataset)
    df["text"] = df["text"].astype(str)
    return df

def split_data(df, train_frac=0.8, random_state=None):
    """
    Splits list of dictionaries (or rows) into 80% train and 20% dev.
    """
    train_df, dev_df = train_test_split(df, test_size=1 - train_frac, random_state=random_state)
    return train_df.reset_index(drop=True), dev_df.reset_index(drop=True)

def log_transform_target(y):
    """
    Apply a base-2 logarithm transformation to the target variable.

    Args:
        y (array-like): Target values.
        offset (float): A small constant added to avoid log(0). Default is 1e-5.

    Returns:
        np.array: Log2-transformed target values.
    """
    y = np.array(y)
    return np.log2(y + 1)

def inverse_log_transform(y):
    y_true = (2 ** y) - 1
    return y_true


def parse_json_file(file_path, gzipped=True):
    """
    Parse a JSON file with one JSON record per line and return a DataFrame.
    
    Args:
        file_path (str): Path to the JSON (or gzipped JSON) file.
        gzipped (bool): If True, the file is expected to be gzipped.
        
    Returns:
        pd.DataFrame: DataFrame created from the JSON records.
    """
    dataset = []
    if gzipped:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            for line in f:
                # Use eval as per provided sample (Note: eval can be unsafe)
                record = eval(line)
                dataset.append(record)
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                record = eval(line)
                dataset.append(record)
    dataset_df = pd.DataFrame(dataset)
    dataset_df["text"] = dataset_df["text"].astype(str)
    return dataset_df["text"]

def get_hours(file_path, gzipped=True):
    """
    Parse a JSON file with one JSON record per line and return a Series of hours.
    
    Args:
        file_path (str): Path to the JSON (or gzipped JSON) file.
        gzipped (bool): If True, the file is expected to be gzipped.
        
    Returns:
        pd.Series: Series containing the 'hours' field from JSON records.
    """
    dataset = []
    if gzipped:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            for line in f:
                record = eval(line)
                dataset.append(record)
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                record = eval(line)
                dataset.append(record)
    dataset_df = pd.DataFrame(dataset)
    dataset_df["hours"] = pd.to_numeric(dataset_df["hours"], errors='coerce')
    return dataset_df["hours"]

def sample_for_testing(df, sample_size=100000, random_state=42):
    """
    Randomly samples rows from a DataFrame for testing.
    """
    return df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

def remove_outliers(df, column="hours", quantile=0.9):
    """
    Removes top quantile rows based on specified column (default: top 10% of hours).
    """
    threshold = df[column].quantile(quantile)
    return df[df[column] <= threshold].reset_index(drop=True)

def reg_evaluate_preds(y_true, y_pred):
    """
    Evaluates predictions with:
    - Mean Squared Error (MSE)
    - Underprediction and Overprediction counts and rates
    """
    mse = mean_squared_error(y_true, y_pred)
    under_rate = np.mean(y_pred < y_true)
    over_rate = np.mean(y_pred > y_true)
    
    return {"MSE": mse, "Underprediction rate": under_rate, "Overprediction rate": over_rate}

def clf_evaluate_preds(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    overpredicts = ((y_pred == 1) & (y_true == 0)).sum()
    underpredicts = ((y_pred == 0) & (y_true == 1)).sum()
    return {"accuracy": accuracy, "Underprediction number": underpredicts, "Overprediction number": overpredicts}

def discretize_hours(df, n_bins=5, strategy='quantile'):
    if 'hours' not in df.columns:
        raise ValueError("DataFrame must contain 'hours' column")
    
    hours = df['hours'].values.reshape(-1, 1)

    discretizer = KBinsDiscretizer(n_bins=n_bins, 
                                 encode='ordinal', 
                                 strategy=strategy)
    df['hours_bin'] = discretizer.fit_transform(hours).astype(int) + 1  # +1 to make bins 1-5
    return df