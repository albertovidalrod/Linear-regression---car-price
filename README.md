# Car price prediction using linear regression

Welcome to my "end-to-end" (still in development) Machine Learning project using linear regression to predict car prices. 

My goal is use this project as a learning experience to understand and implement the different steps in the MLOps framework.

## Description

Originally, I started working on this project to learn how to deploy a simple Machine Learning model. I used Streamlit to deploy the model and after seeing how simple it was, I thought that I could expand the scope of the project. 

This project is now an end-to-end Machine Learning project where new car data is scraped monthly and the model is automatically retrained using the new and old data and then deployed to an app hosted on Streamlit.

You will find in this repository several Jupyter notebooks in which I analyse in detail how linear regression performs on different datasets. This is the data science side of the project, but it is the first step I took.


## Repository structure

    .
    ├── .github                       # yml files for GitHub actions
    ├── data                          # Car data to train the model
    ├── data scraping scripts         # Scripts to scrape new data
    ├── linear regression notebooks   # Notebooks on linear regression on different datasets
    ├── media                         # Media files
    ├── model                         # Linear regression model and script to train it
    ├── app.py                        # Script for the Streamlit app
    └── utils.py                      # Support classes and functions for app.py

The rest of the files are used to create the Docker volume for local development of the Streamlit app

## Model training
The model is trained using the files in the `data` folder:

    .
    ├── ...
    ├── data                    # Car data to train the model
    │   ├── App data        # Data scraped from Exchange and Mart
    │   ├── UK used cars        # Dataset from Kaggle used to train the model
    │   └── autos.csv           # Dataset used in a .ipynb notebook - Not used for the mode
    └── ...
Currently, the `App data` isn't present in the repository because I want to use to trigger the automatic training of the model using GitHub actions. It's currently saved locally, but it will be committed soon.

The linear regression model is created using `train.py`, which can be found in the `model` folder. This script generates the following files:

    .
    ├── ...
    ├── model                                        # Linear regression model and script to train it
    │   │   ├── model_data                           # Output files after training model
    │   │   │   ├── car-price-vX.joblib              # Linear regression model and coefficients
    │   │   │   ├── car-price-vX_metadata.json       # Model metadata
    │   │   │   ├── clean_data-vX.parquet            # Clean data used for model training and in app.py
    │   │   │   ├── data-transformer-vX.joblib       # Transform data before feeding it to model
    │   │   │   └── sample_data-vX.parquet           # Sample data used to fit transformer
    │   ├── generate_clean_data.py                   # Clean data for model using scraped data and Kaggle data
    │   ├── test_clean_data.py                       # Unit test around new clean data
    │   └── train.py                                 # Train new model using new clean data
    └── ...


## Model deployment
Model deployment was my ultimate goal when I started the project. Since this is my first attempt at deploying a Machine Learning model, I chose Streamlit because they make it very easy to create a decent web app with a few lines of code. The app is available on Streamlit cloud: [car price prediction app](https://linear-regression-car-price.streamlit.app). The app is fairly straightforward, as you can see:

![app gif](https://github.com/albertovidalrod/Linear-regression-car-price/blob/main/media/app_example.gif)

The files used to run the app are:

    .
    ├── app.py                        # Script for the Streamlit app
    └── utils.py                      # Support classes and functions for app.py


I used Docker for the development and testing of the web app. As you can see, there are several Docker-related files that I've used to set up my development container:

    .
    ├── .dockerignore                 # Ignore files during COPY operations
    ├── docker-compose.yaml           # Docker compose instructions
    ├── dockerfile                    # Docker image instructions
    └── requirements.txt              # Python packages for Docker image

Streamlit also uses the `requirements.txt` file to install packages, although the python version is selected is the advanced settings when the app is first created. 



## CI / CD
Once I achieved model deployment, I wanted to automate the deployment of future model version using GitHub actions. The file `retrain_model.yml` incldues the necessary steps to generate new data, perform tests to make sure the data format is correct for the model and retrain the model using new data. This action is triggered when new scraped data is committed to the repo, achieving full automation of the process.

## Version History

* v1: initial model version
* v2 (minor update): correct new model to use more robust metrics for model training
* v3: new model using data scraped in June 2023

Planned updates:
* Update model to include option to predict price of electric cars - currently only hybrid, petrol and fuel cars.

## License
This project is licensed under the MIT License - see the LICENSE.md file for details
