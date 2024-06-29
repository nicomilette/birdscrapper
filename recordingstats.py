import pandas as pd
import matplotlib.pyplot as plt

# Load your CSV file
file_path = './bird_recordings.csv'
bird_data = pd.read_csv(file_path)

# Calculate the number of recordings for each bird
recordings_per_bird = bird_data['common_name'].value_counts()

species_less_than_10_recordings = recordings_per_bird[recordings_per_bird < 10].count()
average_recordings = recordings_per_bird.mean()
lowest_recordings = recordings_per_bird.min()
highest_recordings = recordings_per_bird.max()

# Create a summary dataframe
summary_stats = {
    'Average Recordings': average_recordings,
    'Lowest Recordings': lowest_recordings,
    'Highest Recordings': highest_recordings,
    'Total Unique Birds': len(recordings_per_bird),
    'Number of species with less than 10 recordings': species_less_than_10_recordings
}

# Display summary statistics
summary_df = pd.DataFrame.from_dict(summary_stats, orient='index', columns=['Value'])
print("Summary Statistics:")
print(summary_df)

# Plot the number of recordings for each bird
plt.figure(figsize=(10, 6))
recordings_per_bird.plot(kind='bar', logy=True)
plt.title('Number of Recordings per Bird Species')
plt.xlabel('Bird Species')
plt.ylabel('Number of Recordings (log scale)')
plt.xticks([])
plt.show()
