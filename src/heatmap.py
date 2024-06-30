import os
import pandas as pd
import matplotlib.pyplot as plt
import config

# Load the CSV file
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, config.CSV_UNFILTERED)
bird_data = pd.read_csv(file_path)

# Check if required columns are present
required_columns = ['longitude', 'latitude']
for column in required_columns:
    if column not in bird_data.columns:
        raise ValueError(f"Column '{column}' is missing from the CSV file")

# Create a scatter plot of the bird recording locations
plt.figure(figsize=(10, 6))
plt.scatter(bird_data['longitude'], bird_data['latitude'], alpha=0.5, s=10, c='blue')
plt.title('Scatter Plot of Bird Recording Locations')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
