import os
import pandas as pd
import matplotlib.pyplot as plt
import config

# Get the current script directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the CSV file using the config variable
file_path = os.path.join(current_dir, config.CSV_FILE_PATH)

# Load the CSV file
bird_data = pd.read_csv(file_path)

# Calculate the number of recordings for each bird
recordings_per_bird = bird_data['common_name'].value_counts()

species_less_than_10_recordings = recordings_per_bird[recordings_per_bird < config.MIN_RECORDINGS_THRESHOLD].count()
average_recordings = recordings_per_bird.mean()
lowest_recordings = recordings_per_bird.min()
highest_recordings = recordings_per_bird.max()

# Additional statistics
species_with_max_latitude = bird_data.loc[bird_data['latitude'].idxmax()]['common_name']
species_with_min_longitude = bird_data.loc[bird_data['longitude'].idxmin()]['common_name']
average_latitude = bird_data['latitude'].mean()
average_longitude = bird_data['longitude'].mean()

# Create a summary dataframe
summary_stats = {
    'Average Recordings': average_recordings,
    'Lowest Recordings': lowest_recordings,
    'Highest Recordings': highest_recordings,
    'Total Unique Birds': len(recordings_per_bird),
    'Number of species with less than 10 recordings': species_less_than_10_recordings,
    'Species with Maximum Latitude': species_with_max_latitude,
    'Species with Minimum Longitude': species_with_min_longitude,
    'Average Latitude': average_latitude,
    'Average Longitude': average_longitude
}

# Display summary statistics
summary_df = pd.DataFrame.from_dict(summary_stats, orient='index', columns=['Value'])
print("Summary Statistics:")
print(summary_df)

# Plot the number of recordings for each bird
plt.figure(figsize=config.FIGURE_SIZE)
recordings_per_bird.plot(kind='bar', logy=True, color=config.BAR_COLOR)
plt.title('Number of Recordings per Bird Species')
plt.xlabel('Bird Species')
plt.ylabel('Number of Recordings (log scale)')
plt.xticks(rotation=config.ROTATION)
plt.tight_layout()
plt.show()
