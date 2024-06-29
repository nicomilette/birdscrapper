import pandas as pd

# Load the dataset
file_path = './bird_recordings.csv'
bird_data = pd.read_csv(file_path)

# Drop rows with missing common names
bird_data_cleaned = bird_data.dropna(subset=['common_name'])

# Count the number of recordings per species
recordings_per_species = bird_data_cleaned['common_name'].value_counts()

# Identify the top species and new max count
top_species = recordings_per_species.idxmax()
top_count = recordings_per_species.max()
new_max_count = recordings_per_species[recordings_per_species != top_count].max()

# Identify species to be excluded due to having too many recordings (excluding the top species)
species_to_exclude = recordings_per_species[recordings_per_species > new_max_count].index.tolist()
species_to_exclude = [species for species in species_to_exclude if species != top_species]

# Filter the dataset to exclude these species
filtered_data_step1 = bird_data_cleaned[~bird_data_cleaned['common_name'].isin(species_to_exclude)]

# Filter out species with fewer than 20 recordings
species_with_sufficient_recordings = recordings_per_species[recordings_per_species >= 20].index
filtered_data = filtered_data_step1[filtered_data_step1['common_name'].isin(species_with_sufficient_recordings)]

# Save the filtered dataset
filtered_data.to_csv('new_filtered_bird_recordings.csv', index=False)
