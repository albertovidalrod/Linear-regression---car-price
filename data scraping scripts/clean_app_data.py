import os
import csv
import re
import json
from datetime import datetime

import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
data_all_dir = os.path.join(current_dir, "..", "data/App data")
os.makedirs(data_all_dir, exist_ok=True)

# Define path to data
previous_months = [
    x for x in os.listdir(data_all_dir) if ("all" not in x) and (".DS" not in x)
]


# Define a custom key function to extract the datetime from the string
def extract_date(date_string):
    return datetime.strptime(date_string, "%B %Y")


# Sort the dates using the custom key
sorted_months = sorted(previous_months, key=extract_date)
latest_month = sorted_months[-1]
data_month_dir = os.path.join(data_all_dir, latest_month)

# empty list to store data
data_list = []

# loop through files in directory
month_files = os.listdir(data_month_dir)
month_files = sorted(month_files)
for file in month_files:
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

# Define information from previous datasets
previous_files = [x for x in os.listdir(data_all_dir) if ".parquet" in x]

if previous_files:
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

    # Concatenate the previous all cars dataset and the one from the new month
    all_data_previous = pd.read_parquet(os.path.join(data_all_dir, latest_file))
    all_data = pd.concat([all_data_previous, data])
    # Drop duplicates
    all_data = all_data.drop_duplicates(subset="carId").reset_index(drop=True)
    data_save = all_data.copy()
else:
    # Define variables in case no previous all cars dataset is found
    latest_file = "No previous file"
    new_version = 1
    data_save = data.copy()


# Define filename and paths to saved
data_save_filename = f"all_scraped_cars-v{new_version}"
csv_path = data_all_dir + f"/{data_save_filename}.csv"
parquet_path = data_all_dir + f"/{data_save_filename}.parquet"
data_save.to_csv(csv_path)
data_save.to_parquet(parquet_path)

# Generate data for metadata file
# Filename
metadata_filename = f"all_scraped_cars-v{new_version}_metadata"
# Current date
current_date = datetime.now()
current_date_string = current_date.strftime("%d/%m/%Y")

metadata = {
    "filename": f"{metadata_filename}.json",
    "creation_date": current_date_string,
    "latest_source_of_data": latest_month,
    "previous_all_data_file": latest_file,
    "version": f"v{new_version}",
    "associated_files": [f"{data_save_filename}.csv", f"{data_save_filename}.parquet"],
    "columns": {column: str(data_save[column].dtype) for column in data_save.columns},
    "dataset_size": {
        "number_rows": data_save.shape[0],
        "number_columns": data_save.shape[1],
    },
}

# Save the metadata to a JSON file
metadata_path = data_all_dir + f"/{metadata_filename}.json"
with open(metadata_path, "w") as file:
    json.dump(metadata, file, indent=4)
