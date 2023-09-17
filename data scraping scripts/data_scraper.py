import argparse
import os
import time
import traceback
from datetime import datetime

import pandas as pd
import yaml
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.select import Select
from selenium.webdriver.support.ui import WebDriverWait

import chromedriver_autoinstaller
from pyvirtualdisplay import Display


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


def save_data(
    brand: str, cars_data: list, postcode: str, car_type: str, data_dir: str
) -> None:
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
    if brand.casefold() == "all makes".casefold():
        search_brand_str = "all_makes"
    else:
        search_brand_str = brand

    if car_type == "all":
        cars_df.to_csv(f"{data_dir}/{search_brand_str}_{postcode}.csv")
    else:
        cars_df.to_csv(f"{data_dir}/{search_brand_str}_{car_type}.csv")


def scrape_car_data(brand: str, postcode: str, car_type: str, data_dir: str) -> None:
    # Define the url from which the data will be scraped
    url = f"https://www.exchangeandmart.co.uk/used-cars-for-sale/{brand}"

    # Get url information
    # driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    # driver.get(url)

    service = Service()
    options = webdriver.ChromeOptions()
    # options.add_argument("--headless")
    options.add_argument("--window-size=1200,1200")
    # options.add_argument("--ignore-certificate-errors")
    driver = webdriver.Chrome(service=service, options=options)
    driver.get(url)
    wait = WebDriverWait(driver, 20)
    time.sleep(1)

    # cookies_element = driver.find_element(
    #     by=By.CSS_SELECTOR, value="""button[title="I Accept"]"""
    # )

    # wait.until(EC.element_to_be_clickable((By.XPATH, """button[title="I Accept"]""")))
    # driver.execute_script("arguments[0].click();", cookies_element)

    # Define the wait element to pause the script until an element is found or ready to
    # be clicked

    # Find the postcode and update button elements
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
    select_sort_element = wait.until(EC.element_to_be_clickable((By.ID, "ddlSortBy")))
    # Create a Select object from the select element
    select_sort = Select(select_sort_element)
    # Select the option by its distance
    select_sort.select_by_value("distance")

    if car_type != "all":
        # Find the select element by its ID
        select_fuel_element = wait.until(EC.element_to_be_clickable((By.ID, "ddFuel")))

        # Create a Select object from the select element
        select_fuel = Select(select_fuel_element)

        # Select the option by its visible text
        select_fuel.select_by_value(car_type)

        update_button_car_type = driver.find_element(
            by=By.XPATH, value="""//*[@id="searchUpdate"]"""
        )

        wait.until(
            EC.element_to_be_clickable((By.XPATH, """//*[@id="searchUpdate"]"""))
        )
        driver.execute_script("arguments[0].click();", update_button_car_type)

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
                print(f"Finished scraping {postcode}")
                driver.quit()
                del driver
                break
        save_data(brand, cars_data, postcode, car_type, data_dir)

    # In case an Exception is thrown, print the Exception and save the data if it's at least
    # 60 % complete
    except Exception as e:
        traceback.print_exc()  # Print the traceback information
    # finally:
    #     # Save the data in 'cars_data' to a file if it's at least 60 % complete
    #     if cars_data and len(cars_data) > 600:
    #         save_data(brand, cars_data, postcode, car_type, data_dir)


if __name__ == "__main__":
    # Start display
    display = Display(visible=0, size=(800, 800))
    display.start()
    # Update chrome
    chromedriver_autoinstaller.install()
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--search_brand",
        type=str,
        choices=["All makes", "audi", "bmw", "volkswagen"],
        help="Specify the brand to be searched",
    )
    parser.add_argument(
        "-c",
        "--car_type",
        type=str,
        choices=["all", "hybrid", "electric"],
        help="Specify the type of car",
    )
    args = parser.parse_args()
    SEARCH_BRAND = args.search_brand
    car_types_str = args.car_type
    car_types = car_types_str.split(",")

    # Get the current month and create a folder to save the data
    DATE_FOLDER = datetime.now().strftime("%B %Y")
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(CURRENT_DIR, "..", f"data/Scraped data/{DATE_FOLDER}")
    os.makedirs(DATA_DIR, exist_ok=True)

    if SEARCH_BRAND.casefold() == "all makes".casefold():
        config_file_path = os.path.join(CURRENT_DIR, "../config/config_all_makes.yml")
    else:
        config_file_path = os.path.join(CURRENT_DIR, "../config/config.yml")

    # Load the configuration from the YAML file
    with open(config_file_path, "r") as config_file:
        search_config = yaml.safe_load(config_file)

    postcode_all = [search_config["POSTCODES"][datetime.now().day]]

    for car_type in car_types:
        print(car_type)
        for postcode in postcode_all:
            scrape_car_data(SEARCH_BRAND, postcode, car_type, DATA_DIR)
            if car_type != "all":
                break
