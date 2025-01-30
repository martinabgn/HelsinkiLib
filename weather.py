import requests
from bs4 import BeautifulSoup

url = "https://www.accuweather.com/en/fi/helsinki/133328/weather-forecast/133328"

# Set headers to mimic a real browser request
headers = {"User-Agent": "Mozilla/5.0"}

# Send a GET request
response = requests.get(url, headers=headers)

# Check if request was successful
if response.status_code == 200:
    soup = BeautifulSoup(response.text, "html.parser")

    temp_element = soup.find("div", class_="temp")
    temperature = temp_element.text.strip() if temp_element else "N/A"

    condition_element = soup.find("div", class_="phrase")
    condition = condition_element.text.strip() if condition_element else "N/A"

    # Print the extracted data
    print(f"Weather Forecast for Helsinki:")
    print(f"Current Temperature: {temperature}")
    print(f"Condition: {condition}")

else:
    print("Failed to retrieve the webpage. Please check the URL or try again later.")
