from bs4 import BeautifulSoup
from requests_html import HTMLSession
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.select import Select
from webdriver_manager.chrome import ChromeDriverManager


import pandas as pd

import time
import traceback


def car_details(car, driver, wait) -> list:
    car_html = str(car)

    # Create a BeautifulSoup object for the car HTML
    car_soup = BeautifulSoup(car_html, "html.parser")

    # Extract the brand, model and car id (used for finding duplicates)
    brand_model_element = car_soup.find("div")
    car_id = brand_model_element.get("adid")
    brand = brand_model_element.get("make")
    brand = brand.lower()
    model = brand_model_element.get("model")

    # Extract the price
    price_element = car_soup.find("span", class_="price--primary")
    price = price_element.get_text(strip=True).split("£")[1].replace(",", "")

    # Get the mileage from the car ad. This is necessary for electric cars
    mileage_element = car_soup.find(
        class_="key-details__item", string=lambda text: text and "Mileage" in text
    )
    mileage = mileage_element.get_text(strip=True).split()[1].replace(",", "")

    # Click on the link of the car results
    link_element = driver.find_element(By.CSS_SELECTOR, f'a[href*="/ad/{car_id}"]')
    driver.execute_script("arguments[0].click();", link_element)
    # Wait until the "ad details" element is present
    try:
        wait.until(EC.presence_of_element_located((By.ID, "adDetails")))
    except:
        print("Timeout waiting for ad details to load")

    # Parse the new website with the car details
    html = driver.page_source
    car_click_soup = BeautifulSoup(html, "html.parser")

    details_div = car_click_soup.find_all("div", {"class": "adDetsItem"})
    # Extract the text from the div elements
    details_list_all = [div.find("span").text.split()[0] for div in details_div]
    # Use only the required information and get rid of other details
    details_list = details_list_all[:5] + details_list_all[7:]

    # Add the mileage to the car details list for electric vehicles
    if "Electric" in details_list:
        details_list = details_list + [mileage]

    # Modify some of the list elements
    # Engine size
    if "cc" in details_list[1]:
        engine_size = details_list[1].split("cc")[0]
        engine_size = str(round(float(engine_size) / 1000, 1))
        details_list[1] = engine_size
    # Make the text lower case to search for "l" instead of having to search for "L"
    # or "l"
    elif "l" in details_list[1].lower():
        engine_size = details_list[1].lower().split("l")[0]
        details_list[1] = engine_size
    else:
        details_list[1] = float("nan")
        print("Wrong data for engine size")
    # Mileage
    details_list[2] = details_list[2].replace(",", "")
    # Transmission for Semi-Auto cars
    if "Semi" in details_list[4]:
        details_list[4] = "Semi-Auto"

    # Add a "None" value to the car details list if the length of the list is less than
    # 6. This helps for the data manipulation using pandas
    if len(details_list) < 6:
        details_list = details_list + ["None"]

    # Define the list containing all the car details stored on individual variables
    # (model, price, brand and car id) and the details list
    combined_list = (
        [model]
        + [details_list[0]]
        + [price]
        + [details_list[4]]
        + [details_list[2]]
        + details_list[3:4]
        + [details_list[5]]
        + [details_list[1]]
        + [brand]
        + [car_id]
    )

    return (combined_list, driver)


def save_data(cars_data: list, postcode: str) -> None:
    # Create a dataframe containing the cars and save it
    cars_df = pd.DataFrame(cars_data)
    # Define and change the column names of the dataframe
    col_names = [
        "model",
        "year",
        "price",
        "transmission",
        "mileage",
        "fuelType",
        "mpg",
        "engineSize",
        "brand",
        "carId",
    ]
    cars_df.columns = col_names
    # Save the dataframe to csv
    cars_df.to_csv(f"../data/Scraped data/May 2023/{SEARCH_BRAND}_{postcode}.csv")


def scrape_car_data(brand: str, postcode: str) -> None:
    # Define the url from which the data will be scraped
    url = f"https://www.exchangeandmart.co.uk/used-cars-for-sale/{brand}"

    # Get url information
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    driver.get(url)

    # Define the wait element to pause the script until an element is found or ready to
    # be clicked
    wait = WebDriverWait(driver, 20)

    # Dind the postcode and update button elements
    postcode_element = driver.find_element(
        by=By.XPATH, value="""//*[@id="txtPostcode"]"""
    )
    update_button = driver.find_element(
        by=By.XPATH, value="""//*[@id="searchUpdate"]"""
    )

    # Fill the postcode
    postcode_element.send_keys(postcode)
    # Wait until the update button is clickable and then click it
    wait.until(EC.element_to_be_clickable((By.XPATH, """//*[@id="searchUpdate"]""")))
    driver.execute_script("arguments[0].click();", update_button)

    # Find the select element to sort the results
    select_element = wait.until(EC.element_to_be_clickable((By.ID, "ddlSortBy")))
    # Create a Select object from the select element
    select = Select(select_element)
    # Select the option by its distance
    select.select_by_value("distance")

    # Press the "next" button to start scraping data in page 2 out 100 of results. For
    # reason, the script doesn't work on page 1, even if the steps are the same.
    next_button = wait.until(
        EC.presence_of_element_located(
            (By.CSS_SELECTOR, "a.empager__button.empager__button--next")
        )
    )
    driver.execute_script("arguments[0].click();", next_button)

    # Define the list to store the car data
    cars_data = []
    try:
        # Scrape data while the "next" button is present. When page 100 is reached, the
        # script will finish
        while True:
            # Get the HTML source of the updated page
            html = driver.page_source
            soup = BeautifulSoup(html, "html.parser")
            # Find all the cars on the current page (10 cars per page)
            cars = soup.find_all("div", {"class": "result-item"})

            # Scrape the information for each of the cars in the page
            for car in cars:
                if car.get("make") and car.get("model"):
                    try:
                        (car_data, driver) = car_details(car, driver, wait)
                        cars_data.append(car_data)
                        time.sleep(0.1)
                        # Go back to the page before iterating over the next car
                        driver.back()
                    except:
                        # Break in case an Timeout error is thrown
                        break

            try:
                # Press the "next" button to switch to a new page
                # Delete the html element in case it isn't updated properly when the
                # next button is pressed
                del html
                next_button = wait.until(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR, "a.empager__button.empager__button--next")
                    )
                )
                driver.execute_script("arguments[0].click();", next_button)
                time.sleep(0.3)
                # Save the current page number. This variable is used for debugging to
                # find a particular car in case an error is thrown
                current_page = driver.current_url
            except:
                print("Finished scraping")
                break
        save_data(cars_data, postcode)

    # In case an Exception is thrown, print the Exception and save the data if it's at least
    # 60 % complete
    except Exception as e:
        traceback.print_exc()  # Print the traceback information
    finally:
        # Save the data in 'cars_data' to a file if it's at least 60 % complete
        if cars_data and len(cars_data) > 600:
            save_data(cars_data, postcode)


# define the url
SEARCH_BRAND = "volkswagen"

postcode_all = [
    "E34JN",  # East London
    "B24QA",  # Birmingham - Audi
    "NR13JU",  # Norwich - Audi
    "SO140YG",  # Southampton
    "BS11JQ",  # Bristol
    "S14PF",  # Sheffield
    "LS28BH",  # Leeds
    "L34AD",  # Bristol
    "NE77DN",  # Newcastle
    "EH12NG",  # Edinburgh
    "HU67RX",  # Hull
    "EX11SG",  # Exeter
    "CB13EW",  # Cambridge
    "CT12EH",  # Canterbury
    "SA11NU",  # Swansea - Audi
    "BT12HB",  # Belfast - Audi
]

# postcode_all = postcode_all[1:3] + postcode_all[-2:]

for postcode in postcode_all:
    scrape_car_data(SEARCH_BRAND, postcode)
