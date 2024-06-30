import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import librosa.display

def create_spectrogram(file_path, output_path):
    y, sr = librosa.load(file_path, sr=None)
    plt.figure(figsize=(10, 4))
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_DB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel')
    plt.axis('off')  # Turn off the axis
    plt.tight_layout(pad=0)  # Remove padding
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# Define the input and output directories
current_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(current_dir, '..', 'bird_recordings')
output_dir = os.path.join(current_dir, '..', 'spectrograms')

# Create the output directory if it doesn't exist
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Get list of .mp3 files in the input directory
files = [f for f in os.listdir(input_dir) if f.endswith('.mp3')]
total_files = len(files)
completed_files = 0

# Check for already processed files
existing_files = set(f.replace('.png', '.mp3') for f in os.listdir(output_dir) if f.endswith('.png'))

for file_name in files:
    if file_name in existing_files:
        print(f"File {file_name} already processed. Skipping conversion.")
        completed_files += 1
        continue
    
    input_path = os.path.join(input_dir, file_name)
    output_path = os.path.join(output_dir, file_name.replace('.mp3', '.png'))
    create_spectrogram(input_path, output_path)
    completed_files += 1
    progress_percentage = (completed_files / total_files) * 100
    print(f"Processed {file_name} ({progress_percentage:.2f}% complete)")

print("All files processed.")
