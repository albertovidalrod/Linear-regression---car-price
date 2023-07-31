import os
import re
from joblib import load
import pandas as pd
import numpy as np


class PredictCarPrice:
    def __init__(self):
        sample_data = pd.read_parquet("model/model_data/sample_data-v1.parquet")
        self.transformer = load("model/model_data/data_transformer-v2.joblib")
        self.transformer.fit(sample_data)
        # Find all the previous models
        previous_models = [
            x
            for x in os.listdir("model/model_data")
            if (".joblib" in x) and ("car" in x)
        ]
        # Sort files based on version number and find the latest file
        sorted_models = sorted(
            previous_models, key=lambda x: int(re.search(r"\d+", x).group())
        )
        latest_model = sorted_models[-1]
        self.model = load("model/model_data" + f"/{latest_model}")

    def transform(self, input_data):
        col_names = [
            "year",
            "price",
            "transmission",
            "mileage",
            "fuelType",
            "mpg",
            "engineSize",
            "brand",
        ]
        data_df = pd.DataFrame(input_data.reshape(1, -1), columns=col_names)
        self.transformed_data = self.transformer.transform(data_df)

        return self.transformed_data

    def predict(self):
        model_data = self.transformed_data[0, 1:]
        self.prediction_log = self.model.predict(model_data.reshape(1, -1))
        self.prediction = int(np.round(np.exp(self.prediction_log)))

        return self.prediction

    def transform_predict(self, input_data):
        self.transform(input_data)
        self.predict()

        return self.prediction

    def assess_error(self, input_data, price):
        self.transform_predict(input_data)
        self.error = price - self.prediction
        return self.error
