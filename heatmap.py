import pandas as pd
import matplotlib.pyplot as plt

# Load your CSV file
file_path = './bird_recordings.csv'
bird_data = pd.read_csv(file_path)

# Create a scatter plot of the bird recording locations
plt.figure(figsize=(10, 6))
plt.scatter(bird_data['longitude'], bird_data['latitude'], alpha=0.5, s=10, c='blue')
plt.title('Scatter Plot of Bird Recording Locations')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
