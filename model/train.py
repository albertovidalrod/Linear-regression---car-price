import csv
import os

import pandas as pd
import numpy as np
from joblib import dump, load

from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
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
path_data = "/Users/albertovidalrodriguez-bobada/Library/Mobile Documents/com~apple~CloudDocs/Python/Projects/Linear regression - car price/data/UK used cars"

# list of files to load
files_to_load = ["audi.csv", "bmw.csv", "vw.csv"]

# empty list to store data
data_list = []

# loop through files in directory
for file in os.listdir(path_data):
    # check if file is in list of files to load
    if file in files_to_load:
        # open file and read data
        with open(os.path.join(path_data, file), newline="") as f:
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
data = data[data["year"] > 2005].reset_index(drop=True)

# Remove outliers based on "price" feature
mask_audi = (data["brand"] == "audi") & (data["price"] < 75000)
mask_vw = (data["brand"] == "vw") & (data["price"] < 60000)
mask_bmw = (data["brand"] == "bmw") & (data["price"] < 80000)
mask_brand = mask_audi | mask_vw | mask_bmw
data = data[mask_brand].reset_index(drop=True)

# Remove outliers based on "mileage" feature
data = data[data["mileage"] < 150000].reset_index(drop=True)

# Remove outliers based on "MPG" feature
data = data[(data["mpg"] > 18) & (data["mpg"] < 200)].reset_index(drop=True)

# Remove outliers based on "engineSize" feature
data = data[(data["engineSize"] > 1) & (data["engineSize"] < 5.2)].reset_index(
    drop=True
)

mask = (data["fuelType"] == "Other") | (data["fuelType"] == "Electric")
data = data[~mask].reset_index(drop=True)

# Drop irrelevant columns
data.drop(columns=["tax", "model"], axis=1, inplace=True)

# Save clean data
data.to_parquet("clean_data.parquet")

# Load sample data to fit data transformer
sample_data = pd.read_parquet("sample_data.parquet")


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
dump(transformer, "data-transformer-v1.joblib")

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

regression_model = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring=["neg_root_mean_squared_error", "r2"],
    refit="neg_root_mean_squared_error",
    cv=folds,
    n_jobs=-1,
    verbose=0,
)

regression_model.fit(X_train, y_train)

dump(regression_model.best_estimator_, os.getcwd() + "/car-price-v1.joblib")
