import csv
import os
import json
import re
from datetime import datetime

import pandas as pd
import numpy as np

# Load data
current_dir = os.path.dirname(os.path.abspath(__file__))  # current dir is model folder
data_kaggle_dir = os.path.join(current_dir, "..", "data/UK used cars")
os.makedirs(data_kaggle_dir, exist_ok=True)

# list of files to load
files_to_load = ["audi.csv", "bmw.csv", "vw.csv"]

# empty list to store data
data_list = []

# loop through files in directory
data_kaggle_files = os.listdir(data_kaggle_dir)
data_kaggle_files = sorted(data_kaggle_files)
for file in data_kaggle_files:
    # check if file is in list of files to load
    if file in files_to_load:
        # open file and read data
        with open(os.path.join(data_kaggle_dir, file), newline="") as f:
            # create csv reader object
            reader = csv.reader(f)
            # iterate over rows in the csv file and add filename to each row
            reader_data = [row + [file.split(".")[0]] for row in reader]
            # store column names and data in data_list
            col_names = reader_data[0]
            data_list.append(reader_data[1:])

# concatenate data from all files into one dataframe
data_kaggle = pd.concat(
    [pd.DataFrame(data_list[0]), pd.DataFrame(data_list[1]), pd.DataFrame(data_list[2])]
).reset_index(drop=True)

# rename last column to "brand"
col_names[-1] = "brand"
data_kaggle.columns = col_names

# Drop tax column as it hasn't been scraped
data_kaggle.drop(columns=["tax"], inplace=True)

# Defined
data_scraped_dir = os.path.join(current_dir, "..", "data/Scraped data")
os.makedirs(data_kaggle_dir, exist_ok=True)
previous_files = [x for x in os.listdir(data_scraped_dir) if ".parquet" in x]

if previous_files:
    sorted_files = sorted(
        previous_files, key=lambda x: int(re.search(r"\d+", x).group())
    )
    latest_file = sorted_files[-1]

    data_scraped = pd.read_parquet(data_scraped_dir + f"/{latest_file}")
    data_scraped.drop(columns=["carId"], inplace=True)

data = pd.concat([data_kaggle, data_scraped])
data.drop_duplicates(ignore_index=True, inplace=True)


# Transform data types after importing data
int_cols = ["year", "price", "mileage"]
float_cols = ["mpg", "engineSize"]

data[int_cols] = data[int_cols].astype(int)
data[float_cols] = data[float_cols].astype(float)

# Drop duplicates
data.drop_duplicates(ignore_index=True, inplace=True)
data.duplicated().value_counts()

# Remove outliers based on "year" feature
lowest_year = 2005
data = data[data["year"] > lowest_year].reset_index(drop=True)

# Remove outliers based on "price" feature
audi_max_price = 75000
vw_max_price = 60000
bmw_max_price = 80000
mask_audi = (data["brand"] == "audi") & (data["price"] < audi_max_price)
mask_vw = (data["brand"] == "vw") & (data["price"] < vw_max_price)
mask_bmw = (data["brand"] == "bmw") & (data["price"] < bmw_max_price)
mask_brand = mask_audi | mask_vw | mask_bmw
data = data[mask_brand].reset_index(drop=True)

# Remove outliers based on "mileage" feature
min_mileage = 1
max_mileage = 150000
data = data[
    (data["mileage"] > min_mileage) & (data["mileage"] < max_mileage)
].reset_index(drop=True)

# Remove outliers based on "MPG" feature
low_mpg = 18
high_mpg = 200
data = data[(data["mpg"] > low_mpg) & (data["mpg"] < high_mpg)].reset_index(drop=True)

# Remove outliers based on "engineSize" feature
low_engine_size = 1.0
high_engine_size = 5.2
data = data[
    (data["engineSize"] > low_engine_size) & (data["engineSize"] < high_engine_size)
].reset_index(drop=True)

# Remove fuel types for which there are very few datapoints
fuel_type_remove = ["Other", "Electric"]
mask = (data["fuelType"] == fuel_type_remove[0]) | (
    data["fuelType"] == fuel_type_remove[1]
)
data = data[~mask].reset_index(drop=True)

# Drop irrelevant columns
data.drop(columns=["model"], axis=1, inplace=True)

# Save clean data
model_data_dir = current_dir + "/model_data"
previous_data_files = [
    x for x in os.listdir(model_data_dir) if (".parquet" in x) and ("clean" in x)
]

if previous_data_files:
    # Sort files based on version number and find the latest file
    sorted_data_files = sorted(
        previous_data_files, key=lambda x: int(re.search(r"\d+", x).group())
    )
    previous_data_file = sorted_data_files[-1]
    # Extract the last version number and generate the new version number
    previous_version_clean_data = int(
        max(
            [file_name.split("v")[1].split(".")[0] for file_name in previous_data_files]
        )
    )
    new_version_clean_data = previous_version_clean_data + 1
else:
    # Define variables in case no previous all cars dataset is found
    previous_data_file = "No previous clean data file"
    new_version_clean_data = 1
clean_data_name = f"clean_data-v{new_version_clean_data}.parquet"
data.to_parquet(model_data_dir + f"/{clean_data_name}")

# Save intermediate data
data_for_metadata = {
    "clean_data_name": clean_data_name,
    "lowest_year": lowest_year,
    "audi_max_price": audi_max_price,
    "bmw_max_price": bmw_max_price,
    "vw_max_price": vw_max_price,
    "min_mileage": min_mileage,
    "max_mileage": max_mileage,
    "low_mpg": low_mpg,
    "high_mpg": high_mpg,
    "low_engine_size": low_engine_size,
    "high_engine_size": high_engine_size,
    "fuel_type_remove": fuel_type_remove,
}

# Save the metadata to a JSON file
metadata_path = model_data_dir + "/intermediate_data.json"
with open(metadata_path, "w") as file:
    json.dump(data_for_metadata, file, indent=4)
