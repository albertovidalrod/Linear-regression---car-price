import os
import csv
import re

import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
data_all_dir = os.path.join(current_dir, "..", "data/Test scraped data")
os.makedirs(data_all_dir, exist_ok=True)

# Define information from previous datasets
previous_files = [x for x in os.listdir(data_all_dir) if ".parquet" in x]

if previous_files:
    sorted_files = sorted(
        previous_files, key=lambda x: int(re.search(r"\d+", x).group())
    )
    latest_file = sorted_files[-1]

    latest_version = int(
        max([file_name.split("v")[1].split(".")[0] for file_name in previous_files])
    )
    new_version = latest_version + 1
else:
    new_version = 1

# Define path to data
data_month_dir = os.path.join(current_dir, "..", "data/Test scraped data/June 2023")

# empty list to store data
data_list = []

# loop through files in directory
for file in os.listdir(data_month_dir):
    # check if file is in list of files to load
    if ".csv" in file:
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

# Change column type of the columns that are meant to be numeric to int and float types
int_cols = ["year", "price", "mileage"]
float_cols = ["mpg", "engineSize"]
data[int_cols] = data[int_cols].astype(int)
data[float_cols] = data[float_cols].astype(float)

# Round engine size to 1 decimal figure
data["engineSize"] = round(data["engineSize"], 1)

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

# Replace the brand "volkswagen" with "vw" as this used in the other dataset
data.loc[data["brand"] == "volkswagen", "brand"] = "vw"

data_save = data.copy()
data_save.drop(columns=["carId"], inplace=True)

csv_path = data_all_dir + f"/all_scraped_cars-v{new_version}.csv"
parquet_path = data_all_dir + f"/all_scraped_cars-v{new_version}.parquet"
data_save.to_csv(csv_path)
data_save.to_parquet(parquet_path)
