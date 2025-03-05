import pandas as pd
from geopy.geocoders import Nominatim
import time

# Read the data
file_path = "data/Cleaned_Library_Data.csv"
df = pd.read_csv(file_path)

# Initialize the geocoder
geolocator = Nominatim(user_agent="library_locator")

# Create new columns to store latitude and longitude
df["lat"] = None
df["lng"] = None

for index, row in df.iterrows():
    address = row["address"]
    try:
        location = geolocator.geocode(address)
        if location:
            df.at[index, "lat"] = location.latitude
            df.at[index, "lng"] = location.longitude
    except Exception as e:
        print(f"Error fetching {address}: {e}")
    time.sleep(1)  # Prevent requests from being sent too quickly and getting restricted

# 保存结果
df.to_csv("libraries_with_coordinates.csv", index=False)
print("Latitude and longitude retrieval complete, data has been saved to libraries_with_coordinates.csv")
