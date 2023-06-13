import os
import re
import pytest

import pandas as pd


@pytest.fixture
def all_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_all_dir = os.path.join(current_dir, "..", "data/Scraped data")
    os.makedirs(data_all_dir, exist_ok=True)

    # Define information from previous datasets
    previous_files = [x for x in os.listdir(data_all_dir) if ".parquet" in x]

    # Sort files based on version number and find the latest file
    sorted_files = sorted(
        previous_files, key=lambda x: int(re.search(r"\d+", x).group())
    )
    latest_file = sorted_files[-1]
    # Extract the last version number and generate the new version number
    latest_version = int(
        max([file_name.split("v")[1].split(".")[0] for file_name in previous_files])
    )
    new_version = latest_version + 1

    data = pd.read_parquet(os.path.join(data_all_dir, latest_file))

    return data


def test_columns(all_data):
    column_names_types = {
        "model": "object",
        "year": "int64",
        "price": "int64",
        "transmission": "object",
        "mileage": "int64",
        "fuelType": "object",
        "mpg": "float64",
        "engineSize": "float64",
        "brand": "object",
        "carId": "object",
    }
    assert {
        column: str(all_data[column].dtype) for column in all_data.columns
    } == column_names_types


def test_brand_values(all_data):
    brand_values = ["audi", "bmw", "vw"]
    assert set(brand_values) == set(all_data["brand"])


def test_fuel_type_values(all_data):
    fuel_type_values = ["Petrol", "Diesel", "Hybrid", "Electric"]
    assert set(fuel_type_values) == set(all_data["fuelType"])


def test_transmission_values(all_data):
    transmission_values = ["Manual", "Automatic", "Semi-Auto"]
    assert set(transmission_values) == set(all_data["transmission"])
