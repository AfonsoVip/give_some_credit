import pandas as pd
import pytest

# Import the read_data function from the nodes.py file to access the raw data
from give_me_some_credit.nodes import read_data

# Read the raw data
raw_data = read_data("data/01_raw/data.csv")

def test_not_empty():
    assert not raw_data.empty, "The raw data is empty."

def test_missing_values():
    missing_values = raw_data.isnull().sum().sum()
    assert missing_values == 0, f"The raw data contains {missing_values} missing values."

def test_positive_values():
    negative_values = (raw_data < 0).sum().sum()
    assert negative_values == 0, f"The raw data contains {negative_values} negative values."

def test_data_type_consistency():
    data_types = raw_data.dtypes
    for column, dtype in data_types.iteritems():
        assert dtype == data_types[0], f"Inconsistent data type in column '{column}'. Expected {data_types[0]}, got {dtype}."