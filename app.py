import streamlit as st
import numpy as np
from utils import PredictCarPrice
import pandas as pd
import pygwalker as pyg
import plotly.graph_objects as go
import numpy as np
import plotly.figure_factory as ff

# Load data for the plot
clean_data = pd.read_parquet("model/clean_data.parquet")


st.title("Estimate the price of a used car")

# INPUT DATA
with st.container():
    # create three columns of equal width
    col_1, col_2, col_3 = st.columns(3)

    # create a selectbox in each column
    with col_1:
        brand = st.selectbox("Select car brand", ["Audi", "BMW", "Volkswagen"])

    with col_2:
        transmission = st.selectbox(
            "Input transmission type", ["Manual", "Semi-Auto", "Automatic"]
        )
    with col_3:
        fuel_type = st.selectbox("Select fuel type: ", ["Petrol", "Diesel", "Hybrid"])

with st.container():
    # create three columns of equal width
    col_1, col_2, col_3 = st.columns(3)

    # create a selectbox in each column
    with col_1:
        mileage = st.number_input(
            "Input car mileage: ", 0, 1500000, value=20000, step=1
        )

    with col_2:
        mpg = st.number_input(
            "Input car mpg: ",
            18.0,
            200.0,
            value=30.0,
            step=0.1,
        )
    with col_3:
        engine_size = st.number_input(
            "Input engine size: ", 1.0, 5.2, value=2.0, step=0.1
        )

year = st.slider("Input year the car was manufactured", 2005, 2020, value=2010)

st.caption(
    """Note: Although *Hybrid* is included as one of the options for the fuel type, 
    there aren't many data samples. These only include Automatic and Semi-Automatic cars
    and cars manufactured from 2014 onwards. Selecting other values
    will lead to wrong predictions as the model will extrapolate"""
)

# Convert brand name to values used by the model. Otherwise, an error will be thrown
if brand == "Audi":
    brand = "audi"
elif brand == "BMW":
    brand = "bmw"
else:
    brand = "vw"


# Define function to make prediction given user input data
def make_prediction():
    input_data = np.array(
        [year, 1, transmission, mileage, fuel_type, mpg, engine_size, brand]
    )

    predictor = PredictCarPrice()
    price = predictor.transform_predict(input_data)

    st.write(f"The estimated price is **£ {price}**")  # Display the prediction result

    return price


placeholder = st.empty()  # Create an empty placeholder
predict_button = st.button("Make prediction")

if st.session_state.get("predict_button") != True:
    st.session_state["predict_button"] = predict_button

if st.session_state["predict_button"] == True:
    price = make_prediction()
    placeholder.empty()  # Remove the button from the placeholder

    # st.write("See how your predictions compare to our database:")

    # Select features for the plot to see how the prediction compares to the database
    with st.container():
        # create two columns of equal width
        col_1, col_2 = st.columns(2)

        # create a selectbox in each column
        with col_1:
            feature_1 = st.selectbox(
                "Select car feature 1", ["mpg", "mileage", "engineSize", "year"]
            )

        with col_2:
            feature_2 = st.selectbox(
                "Select car feature 2", ["mileage", "mpg", "engineSize", "year"]
            )

    plot_button = st.button("Generate plot")

    if st.session_state.get("plot_button") != True:
        st.session_state["plot_button"] = plot_button

    # Generate graph if the button is pressed
    if st.session_state["plot_button"] == True:
        data_brand = clean_data[clean_data["brand"] == brand]
        fig = go.Figure()

        # Add trace containing database. Feature 1 is plotted in the y-axis and feature
        # 2 is represented using the color of the marker
        fig.add_trace(
            go.Scatter(
                x=data_brand[feature_1],
                y=data_brand["price"],
                marker=dict(
                    color=data_brand[feature_2],
                    colorscale="Viridis",
                    colorbar=dict(title=feature_2),
                    showscale=True,
                ),
                mode="markers",
                name="Data",
            )
        )
        # Add trace representing the prediction
        fig.add_trace(
            go.Scatter(
                x=[eval(feature_1)],
                y=[price],
                marker=dict(symbol="diamond", color="red", size=10),
                mode="markers",
                name="Prediction",
            )
        )

        # Set location and orientation of the legend to prevent it from overlapping with
        # the color bar
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font_size=14,
            ),
            xaxis=dict(title=feature_1, title_font=dict(size=16)),
            yaxis=dict(title="Price (£)", title_font=dict(size=16)),
            title=dict(
                text="Comparison of prediction vs database (only cars of the same brand)",
                font=dict(size=20),
            ),
        )

        st.plotly_chart(fig, use_container_width=True)
        st.caption("Note: the colour of the data points represents the second feature")

        # with st.container():
        #     # create two columns of equal width
        #     col_1, col_2 = st.columns(2)

        #     # create a selectbox in each column
        #     with col_1:
        #         if st.button("Make new prediction"):
        #             st.session_state["predict_button"] = False
        #             st.session_state["plot_button"] = False
        #             st.checkbox("Click to make new prediction")

        #     with col_2:
        #         if st.button("Choose new features for the plot"):
        #             st.session_state["plot_button"] = False
        #             st.checkbox("Click to choose new features for the plot")
