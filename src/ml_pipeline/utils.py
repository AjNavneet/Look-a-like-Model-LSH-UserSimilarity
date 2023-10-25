import numpy as np
import pandas as pd
from itertools import chain
from collections import Counter
from yaml import CLoader as Loader, load

# Function to read data from a CSV file
def read_data_csv(file_path, **kwargs):
    """
    Read data from a CSV file and return it as a DataFrame.

    :param file_path: Path to the CSV file
    :param **kwargs: Additional keyword arguments for pandas.read_csv()
    :return: DataFrame containing the data
    """
    raw_data_csv = pd.read_csv(file_path, **kwargs)
    return raw_data_csv

# Function to read data from a JSON file
def read_data_json(file_path, **kwargs):
    """
    Read data from a JSON file and return it as a DataFrame.

    :param file_path: Path to the JSON file
    :param **kwargs: Additional keyword arguments for pandas.read_json()
    :return: DataFrame containing the data
    """
    raw_data_json = pd.read_json(file_path, **kwargs)
    return raw_data_json

# Function for reading a YAML configuration file
def read_config(path):
    """
    Read a YAML configuration file and return it as a dictionary.

    :param path: Path to the YAML configuration file
    :return: Dictionary containing the configuration settings
    """
    with open(path) as stream:
        config = load(stream, Loader=Loader)
    return config

# Calculate feature importance (i.e. ranking the users) based on probabilities
def feat_imp(p, q):
    """
    Calculate feature importance given seed set probability of a feature and global
    probability for the same feature.

    :param p: Probability in the seed set
    :param q: Global probability
    :return: Feature importance score
    """
    return (p - q) * np.log((p * (1 - q)) / ((1 - p) * q))

# Calculate the counts of features in the dataset
def count_fn(data, features, list_cols):
    """
    Calculate the count of features in the dataset and their probabilities.

    :param data: DataFrame containing user records
    :param features: List of columns in the dataset
    :param list_cols: Columns that can have multiple values per user
    :return: DataFrame with feature count values and probabilities
    """
    count_df = pd.DataFrame(columns=["value", "count", "feature"])
    for col in features:
        if col not in list_cols:
            counts = data[col].value_counts()
        else:
            counts = pd.Series(Counter(chain.from_iterable(x for x in data[col])))
        counts = counts.reset_index()
        counts.columns = ["value", "count"]
        counts["feature"] = col
        count_df = pd.concat([count_df, counts])
    count_df["sum"] = count_df.groupby("feature")["count"].transform(sum)
    count_df["prob"] = count_df["count"] / count_df["sum"]
    count_df = count_df[["feature", "value", "prob"]]
    return count_df

# Convert column values into strings with column name as a prefix
def conv_values(v, c, list_c):
    """
    Convert column values into strings with the column name as a prefix.

    :param v: Feature value
    :param c: Column name
    :param list_c: Boolean, True if c is a list column, False otherwise
    :return: List of string-prefixed feature values or a single string
    """
    if list_c:
        final_v = []
        for v_ in v:
            final_v.append(f"{c}_{str(v_)}")
    else:
        final_v = f"{c}_{str(v)}"
    return final_v

# Flatten a list of lists
def flatten_list(f):
    """
    Flatten a list of lists.

    :param f: List of lists
    :return: A flat list
    """
    f_l = []
    for f_ in f:
        if isinstance(f_, list):
            f_l.extend(f_)
        else:
            f_l.append(f_)
    return f_l
