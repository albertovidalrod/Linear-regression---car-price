# Car price prediction using linear regression

Welcome to my "end-to-end" (still in development) Machine Learning project using linear regression to predict car prices. 

My goal is use this project as a learning experience to understand and implement the different steps in the MLOps framework.

## Description

Originally, I started working on this project to learn how to deploy a simple Machine Learning model. I used Streamlit to deploy the model and after seeing how simple it was, I thought that I could expand the scope of the project.

The current goal of the project is to use a web scraper to gather more data, trigger the re-training of the model, which in itself triggers the CI / CD pipeline and finally the new model is deployed. 

The project isn't there yet, but I've taken a first few steps in the right direction:
- [x] Create v1 of the model using Kaggle's data
- [x] Develop Streamlit app locally using Docker
- [x] Develop a web scraper to gather more data

But there are a few tasks to complete:
- [ ] Deploy streamlit app
- [ ] Automatically re-train model after new data is gathered
- [ ] CI: write unit testing
- [ ] CI / CD: deploy new model after re-training and test that there are no errors

Although the open actions are challenging, I believe that I've already completed the most daunting tasks!

You will find in this repository several Jupyter notebooks in which I analyse in detail how linear regression performs on different datasets. This is the data science side of the project, but it is the first step I took.


## Repository structure

    .
    ├── data                          # Car data to train the model
    ├── data scraping scripts         # Scripts to scrape new data
    ├── linear regression notebooks   # Notebooks on linear regression on different datasets
    ├── model                         # Linear regression model and script to train it
    ├── app.py                        # Script for the Streamlit app
    └── utils.py                      # Support classes and functions for app.py

The rest of the files are used to create the Docker volume for local development of the Streamlit app

## Model training
The model is trained using the files in the `data` folder:

    .
    ├── ...
    ├── data                    # Car data to train the model
    │   ├── Scraped data        # Data scraped from Exchange and Mart
    │   ├── UK used cars        # Dataset from Kaggle used to train the model
    │   └── autos.csv           # Dataset used in a .ipynb notebook - Not used for the mode
    └── ...
Currently, the `Scraped data` isn't present in the repository because I want to use to trigger the automatic training of the model using GitHub actions. It's currently saved locally, but it will be committed soon.

The linear regression model is created using `train.py`, which can be found in the `model` folder. This script generates the following files:

    .
    ├── ...
    ├── model                             # Linear regression model and script to train it
    │   ├── car-price-vX.joblib           # Linear regression model and coefficients
    │   ├── clean_data.parquet            # Clean data used for model training and in app.py
    │   ├── data-transformer-vX.joblib    # Transform data before feeding it to model
    │   └── sample_data.parquet           # Sample data used to fit transformer
    └── ...

Where X represents the version of the model.

- [ ] Create model metadata file to keep track of changes
## Model deployment
Model deployment was my ultimate goal when I started the project. Since this is my first attempt at deploying a Machine Learning model, I chose Streamlit because they make it very easy to create a decent web app with a few lines of code. The files used to run the app are:

    .
    ├── app.py                        # Script for the Streamlit app
    └── utils.py                      # Support classes and functions for app.py


I used Docker for the development and testing of the web app. As you can see, there are several Docker-related files that I've used to set up my development container:

    .
    ├── .dockerignore                 # Ignore files during COPY operations
    ├── docker-compose.yaml           # Docker compose instructions
    ├── dockerfile                    # Docker image instructions
    └── requirements.txt              # Python packages for Docker image


The app runs as desired, but I'm yet to publish it in Streamlit. 

## MLOps

This project offers a great opportunity to implement more elements of the MLOps framework such automatic training or CI/CD. 

This elements are yet to be implemented, but I will use GitHub actions to do it.

## Version History

*To be updated*

## License
*To be updated*

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details