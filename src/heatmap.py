import webbrowser
import os

import os
import pandas as pd
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
import config

current_dir = os.path.dirname(os.path.abspath(__file__))


# Load the CSV file
print("Generating heatmap...")

file_path = os.path.join(current_dir, config.CSV_FILE_PATH_HEATMAP)
bird_data = pd.read_csv(file_path)

# Check for missing latitude and longitude and drop those rows
bird_data = bird_data.dropna(subset=['latitude', 'longitude'])

# Create a base map centered at the mean latitude and longitude
map_center = [bird_data['latitude'].mean(), bird_data['longitude'].mean()]
base_map = folium.Map(location=map_center, zoom_start=5)

# Add a heat map layer
heat_data = [[row['latitude'], row['longitude']] for index, row in bird_data.iterrows()]
HeatMap(heat_data).add_to(base_map)

# Save the map as an HTML file

html_file_path = os.path.join(current_dir, '../heatmaps/bird_recordings_heatmap.html')
directory = os.path.dirname(html_file_path)
if directory:
    os.makedirs(directory, exist_ok=True)  # Ensure the directory exists

base_map.save(html_file_path)


print("Opening heatmap...")
# Define the path to the HTML file
# Get the current script directory
# Construct the full path to the CSV file using the config variable

# Ensure the file exists
if os.path.exists(html_file_path):
    # Open the HTML file in the default web browser
    webbrowser.open(f'file://{os.path.abspath(html_file_path)}')
else:
    print(f"The file {html_file_path} does not exist.")
