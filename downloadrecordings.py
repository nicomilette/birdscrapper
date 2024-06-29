import os
import pandas as pd
import requests
from urllib.parse import urlparse

# Load the filtered dataset
file_path = './new_filtered_bird_recordings.csv'  # Update this path to your filtered dataset
bird_data = pd.read_csv(file_path)

# Directory to save the recordings
output_dir = 'bird_recordings'
os.makedirs(output_dir, exist_ok=True)

# Function to clean file name
def clean_filename(name):
    return "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in name)

# Function to download a recording
def download_recording(recording_url, output_path):
    try:
        response = requests.get(recording_url, stream=True)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
        else:
            print(f"Failed to download {recording_url}")
    except Exception as e:
        print(f"Error downloading {recording_url}: {e}")

# Total number of recordings
total_recordings = bird_data.shape[0]
downloaded_recordings = 0

# Download all recordings
for index, row in bird_data.iterrows():
    recording_url = row['recording_url']
    recording_id = row['id']
    common_name = row['common_name']
    file_name = f"{clean_filename(common_name)}_{recording_id}.mp3"
    output_path = os.path.join(output_dir, file_name)
    
    # Check if the file already exists
    if os.path.exists(output_path):
        print(f"File {file_name} already exists. Skipping download.")
        downloaded_recordings += 1
        continue

    download_recording(recording_url, output_path)
    downloaded_recordings += 1
    progress_percentage = (downloaded_recordings / total_recordings) * 100
    print(f"Downloaded {file_name} \n({progress_percentage:.2f}% complete)")

print("All recordings downloaded.")
