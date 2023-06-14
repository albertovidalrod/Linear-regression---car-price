import csv
import os
import json
from datetime import datetime

import pandas as pd
import numpy as np
from joblib import dump, load

from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV,
    train_test_split,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    PolynomialFeatures,
    OneHotEncoder,
    FunctionTransformer,
    MinMaxScaler,
)


# Load data
current_dir = os.path.dirname(os.path.abspath(__file__))  # current dir is model folder
data_dir = os.path.join(current_dir, "..", "data/UK used cars")
os.makedirs(data_dir, exist_ok=True)

# list of files to load
files_to_load = ["audi.csv", "bmw.csv", "vw.csv"]

# empty list to store data
data_list = []

# loop through files in directory
for file in os.listdir(data_dir):
    # check if file is in list of files to load
    if file in files_to_load:
        # open file and read data
        with open(os.path.join(data_dir, file), newline="") as f:
            # create csv reader object
            reader = csv.reader(f)
            # iterate over rows in the csv file and add filename to each row
            reader_data = [row + [file.split(".")[0]] for row in reader]
            # store column names and data in data_list
            col_names = reader_data[0]
            data_list.append(reader_data[1:])

# concatenate data from all files into one dataframe
data = pd.concat(
    [pd.DataFrame(data_list[0]), pd.DataFrame(data_list[1]), pd.DataFrame(data_list[2])]
).reset_index(drop=True)

# rename last column to "brand"
col_names[-1] = "brand"
data.columns = col_names

# Transform data types after importing data
int_cols = ["year", "price", "mileage", "tax"]
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
max_mileage = 150000
data = data[data["mileage"] < 150000].reset_index(drop=True)

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
data.drop(columns=["tax", "model"], axis=1, inplace=True)

# Save clean data
new_version_clean_data = 1
clean_data_name = f"clean_data-v{new_version_clean_data}.parquet"
data.to_parquet(current_dir + f"/{clean_data_name}")

# Load sample data to fit data transformer
new_version_sample_data = 1
sample_data_name = f"sample_data-v{new_version_sample_data}.parquet"
sample_data = pd.read_parquet(current_dir + f"/{sample_data_name}")


# Define features to be one-hot-encoded, log transformed and non-transformed
ohe_cols = ["transmission", "fuelType", "brand"]
log_cols = ["price", "mileage"]
log_cols_transformed = [column + "_log" for column in log_cols]
non_transformed_cols = [
    column
    for column in data.columns.tolist()
    if (column not in ohe_cols) & (column not in log_cols)
]

# Create data transformer. Note that these
log_transformer = FunctionTransformer(func=np.log, inverse_func=np.exp, validate=True)
transformer = make_column_transformer(
    (log_transformer, log_cols),
    (OneHotEncoder(drop="first"), ohe_cols),
    remainder="passthrough",
)

# Transform data
transformer.fit(sample_data)
transformed = transformer.transform(data)

# Save the transformer to a file
new_version_transformer = 1
transformer_name = f"data_transformer-v{new_version_transformer}.joblib"
dump(transformer, current_dir + f"/{transformer_name}")


# Define column names of new columns created after one-hot-encoding transformation
ohe_cols_transformed = (
    transformer.named_transformers_["onehotencoder"].get_feature_names_out().tolist()
)
# Define the name of all the new columns
all_transformed_cols = (
    log_cols_transformed + ohe_cols_transformed + non_transformed_cols
)

# Define index of column containing fuelType_Hybrid data. It will be used in
# train_test_split as the stratified variable
hybrid_idx = (log_cols + ohe_cols_transformed).index("fuelType_Hybrid")

X_data = transformed[:, 1:]
y_data = transformed[:, 0]

X_train, X_test, y_train, y_test = train_test_split(
    X_data,
    y_data,
    test_size=0.25,
    random_state=42,
    shuffle=True,
    stratify=X_data[:, hybrid_idx - 1],
)

num_splits = 5
sk_fold = StratifiedKFold(n_splits=num_splits, random_state=42, shuffle=True)
folds = sk_fold.split(X_train, X_train[:, hybrid_idx])

# Define pipeline steps
steps = [
    ("scaler", MinMaxScaler()),
    ("pol_features", PolynomialFeatures()),
    ("model", LinearRegression()),
]

# Create pipeline object
pipeline = Pipeline(steps=steps)

param_grid = {"pol_features__degree": range(1, 6)}


# Define metric for Grid Search process
# The rmse of the log transformed prices isn't the same as the rmse of the actual prices
# Therefore, I will create a metric that computes the rmse of the actual prices and
# predictions to select the best performing model
def rmse(y_true, y_pred):
    # Compute the rmse of log prices
    rmse_log = np.sqrt(mean_squared_error(y_true, y_pred))
    rmse_value = rmse_log
    #  Sometimes, if the degree is too high for an accurate model, the rmse will yield
    # cray values exceeded 10000 (for reference, for the log prices the average rmse is
    # 0.2)
    # If the value is less than 1000, I will compute the rmse of the actual prices.
    # Otherwise, the output of the function is the rmse of log prices
    if rmse_log < 1000:
        rmse_exp = np.sqrt(mean_squared_error(np.exp(y_true), np.exp(y_pred)))
        rmse_value = rmse_exp
    return rmse_value


rmse_scorer = make_scorer(rmse, greater_is_better=False)
scoring_params = rmse_scorer
scoring_params_name = "rmse_function"
regression_model = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring=scoring_params,
    cv=folds,
    n_jobs=-1,
    verbose=0,
)

regression_model.fit(X_train, y_train)
# Save model
new_version_model = 2
model_name = f"car_price-v{new_version_model}"
dump(regression_model.best_estimator_, current_dir + f"/{model_name}.joblib")

# Make predictions and
predictions_log = regression_model.best_estimator_.predict(X_test)
predictions = np.exp(predictions_log)
y_test_exp = np.exp(y_test)
model_rmse = np.sqrt(mean_squared_error(y_test_exp, predictions))
model_r2_score_log = r2_score(y_test, predictions_log)
model_r2_score = r2_score(y_test_exp, predictions)


# Generate data for metadata file
# Filename
# metadata_filename = f"all_scraped_cars-v{new_version}_metadata"
metadata_filename = f"{model_name}_metadata"
# Current date
current_date = datetime.now()
current_date_string = current_date.strftime("%d/%m/%Y")

metadata = {
    "filename": f"{metadata_filename}.json",
    "creation_date": current_date_string,
    "source_of_data": ["autos"],
    "version": f"v{new_version_model}",
    "associated_model": [
        f"{model_name}.joblib",
        transformer_name,
        clean_data_name,
        sample_data_name,
    ],
    "model_polynomial_degree": regression_model.best_estimator_.named_steps[
        "pol_features"
    ].degree,
    "model_metrics": {
        "rmse": f"{model_rmse:.2f}",
        "r2_score": f"{model_r2_score:.3f}",
        "r2_score_log_predictions": f"{model_r2_score_log:.3f}",
    },
    "model_scoring": scoring_params_name,
    "outlier_limits": {
        "lowest_year": lowest_year,
        "car_max_prices": {
            "audi": audi_max_price,
            "bmw": bmw_max_price,
            "volkswagen": vw_max_price,
        },
        "max_mileage": max_mileage,
        "mpg_limits": {"low": low_mpg, "high": high_mpg},
        "engine_size_limits": {"low": low_engine_size, "high": high_engine_size},
        "fuel_type_removed": fuel_type_remove,
    },
}

# Save the metadata to a JSON file
metadata_path = current_dir + f"/{metadata_filename}.json"
with open(metadata_path, "w") as file:
    json.dump(metadata, file, indent=4)
