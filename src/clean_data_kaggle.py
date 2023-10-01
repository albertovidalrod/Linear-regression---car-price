import os
import csv
import re
import json
from datetime import datetime

import pandas as pd


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ALL_DIR = os.path.join(CURRENT_DIR, "..", "data/Scraped data")
DATA_KAGGLE_DIR = os.path.join(CURRENT_DIR, "..", "data/Kaggle data")
os.makedirs(DATA_ALL_DIR, exist_ok=True)
os.makedirs(DATA_KAGGLE_DIR, exist_ok=True)

car_names = [
    "bmw",
    "volkswagen",
    "audi",
    "vauxhall",
    "ford",
    "mercedes-benz",
    "kia",
    "nissan",
    "citroen",
    "renault",
    "toyota",
    "peugeot",
    "skoda",
    "seat",
    "mini",
]
for car_name in car_names:
    os.makedirs(os.path.join(DATA_KAGGLE_DIR, car_name), exist_ok=True)

# Define path to data
previous_months = [
    x for x in os.listdir(DATA_ALL_DIR) if (".csv" not in x) and (".DS" not in x)
]


# Define a custom key function to extract the datetime from the string
def extract_date(date_string):
    return datetime.strptime(date_string, "%B %Y")


# Sort the dates using the custom key
sorted_months = sorted(previous_months, key=extract_date)
latest_month = sorted_months[-1]
data_month_dir = os.path.join(DATA_ALL_DIR, latest_month)

month_files = [
    file
    for file in os.listdir(data_month_dir)
    if (".csv" in file) and ("all_makes" in file)
]
month_files = sorted(month_files)

# empty list to store data
data_list = []
for file in month_files:
    # open file and read data
    with open(os.path.join(data_month_dir, file), newline="") as f:
        # create csv reader object
        reader = csv.reader(f)
        # iterate over rows in the csv file and add filename to each row
        reader_data = [row + [file.split(".")[0]] for row in reader]
        # store column names and data in data_list
        col_names = reader_data[0]
        data_list.append(reader_data[1:])

# concatenate data from all files into one dataframe
data = pd.concat([pd.DataFrame(data) for data in data_list]).reset_index(drop=True)

# Drop the first (index as a column) and last column (postcode - the above code must
# have added it)
data.drop(columns=[0, 11], inplace=True)
data.columns = col_names[1:11]

# Drop duplicates
data = data.drop_duplicates(subset="carId").reset_index(drop=True)

# Remove cars of fuel type equal to 'Petrol/Electric'
data = data[data["fuelType"] != "Petrol/Electric"]
data.reset_index(drop=True, inplace=True)

# Create a boolean ev_mask for elements containing the word "Electric"
ev_mask = data.apply(lambda x: x.str.contains("Electric", case=False)).any(axis=1)
ev_mileage = data.loc[ev_mask, "mpg"]
data = data.copy()
data.loc[ev_mask, "mileage"] = ev_mileage

# Clean data from electric cars. Because some fields such as mpg or engineSize doesn't
# really apply to them, the scraping tool didn't get the information right
data.loc[ev_mask, "transmission"] = "Automatic"
data.loc[ev_mask, "mpg"] = 0
data.loc[ev_mask, "fuelType"] = "Electric"
data.loc[ev_mask, "engineSize"] = 0

# Check if previous data is available in the Kaggle data directory
previous_files = [x for x in os.listdir(DATA_KAGGLE_DIR) if "all_cars" in x]

if previous_files:
    previous_data = pd.read_csv(f"{DATA_KAGGLE_DIR}/all_data.csv")
    data = pd.concat([data, previous_data])
    data = data.drop_duplicates(subset="carId").reset_index(drop=True)

# Save the new version of the non-cleaned data
data.to_csv(f"{DATA_KAGGLE_DIR}/all_data.csv", index=False)

# Save each of the brands
for car_name in car_names:
    data_brand = data[data["brand"] == car_name]
    data_brand = data.reset_index(drop=True)
    data_brand.to_csv(f"{DATA_KAGGLE_DIR}/{car_name}/{car_name}.csv")

# Turn the mileage column into numeric type and remove NaN values
data["mileage"] = pd.to_numeric(data["mileage"], errors="coerce")
data = data.dropna()
data.reset_index(drop=True, inplace=True)

# Create a mask to identify where a value in any columns is equal to "None"
columns_to_apply = data.columns[data.columns != "mileage"]
none_mask = (
    data[columns_to_apply]
    .apply(lambda x: x.str.contains("None", case=False))
    .any(axis=1)
)
# Create a mask to identify where the mileage column is an integer - float values need
# be removed
int_mask = data["mileage"].apply(lambda x: x.is_integer())

# Apply the masks to remove float values in mileage and any values equal to "None"
data = data[(~none_mask) & (int_mask)]
data.reset_index(drop=True, inplace=True)

# Make sure that categorical columns only contain the right categories
# Fuel type
data = data.loc[
    (data["fuelType"] == "Petrol")
    | (data["fuelType"] == "Diesel")
    | (data["fuelType"] == "Hybrid")
    | (data["fuelType"] == "Electric")
]
data.reset_index(drop=True, inplace=True)
# Transmission
data = data.loc[
    (data["transmission"] == "Automatic")
    | (data["transmission"] == "Manual")
    | (data["transmission"] == "Semi-Auto")
]
data.reset_index(drop=True, inplace=True)

# Change column type of the columns that are meant to be numeric to int and float types
int_cols = ["year", "price", "mileage"]
float_cols = ["mpg", "engineSize"]
data[int_cols] = data[int_cols].astype(int)
data[float_cols] = data[float_cols].astype(float)

# Round engine size to 1 decimal figure
data["engineSize"] = round(data["engineSize"], 1)


# Save the new version of the non-cleaned data
data.to_csv(f"{DATA_KAGGLE_DIR}/all_data_cleaned.csv", index=False)

# Save each of the brands
for car_name in car_names:
    data_brand = data[data["brand"] == car_name]
    data_brand = data.reset_index(drop=True)
    data_brand.to_csv(f"{DATA_KAGGLE_DIR}/{car_name}/{car_name}_cleaned.csv")
