from joblib import load
import pandas as pd
import numpy as np


class PredictCarPrice:
    def __init__(self):
        sample_data = pd.read_parquet("model/sample_data.parquet")
        self.transformer = load("model/data-transformer-v1.joblib")
        self.transformer.fit(sample_data)
        self.model = load("model/car-price-v1.joblib")

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
