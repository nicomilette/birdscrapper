import os
import pandas as pd
import config

def filter_recordings(data):
    # Remove rows with missing common names
    filtered_data = data.dropna(subset=['common_name'])

    # Filter out entries with the common name "identity unknown"
    filtered_data = filtered_data[
        ~filtered_data['common_name'].str.lower().isin(['identity unknown', 'soundscape'])
    ]

    # Filter out species with less than 20 recordings
    species_recordings = filtered_data['scientific_name'].value_counts()
    species_to_keep = species_recordings[species_recordings >= 20].index
    filtered_data = filtered_data[filtered_data['scientific_name'].isin(species_to_keep)]

    return filtered_data

def save_to_csv(data, filename):
    if data.empty:
        print("No data to save.")
        return

    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)  # Ensure the directory exists

    data.to_csv(filename, index=False, encoding='utf-8')
    print(f"Data saved to {filename}")

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    file_path = os.path.join(current_dir, '../tables/bird_recordings_unfiltered.csv')

    # Load the unfiltered data
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    bird_data = pd.read_csv(file_path)

    # Filter the data
    filtered_data = filter_recordings(bird_data)

    output_path = os.path.join(current_dir, '../tables/bird_recordings_filtered.csv')

    # Save the filtered data to the new CSV file
    save_to_csv(filtered_data, output_path)

if __name__ == "__main__":
    main()
