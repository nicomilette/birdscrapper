import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import librosa.display
from multiprocessing import Pool, cpu_count, Manager

def create_spectrogram(file_info):
    file_name, input_dir, output_dir, progress, total_files = file_info
    input_path = os.path.join(input_dir, file_name)
    output_path = os.path.join(output_dir, file_name.replace('.mp3', '.png'))

    if os.path.exists(output_path):
        print(f"File {file_name} already processed. Skipping conversion.")
        progress['completed'] += 1
        progress_percentage = (progress['completed'] / total_files) * 100
        print(f"Processed {file_name} ({progress_percentage:.2f}% complete)")
        return file_name, False
    
    y, sr = librosa.load(input_path, sr=None)
    plt.figure(figsize=(10, 4))
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_DB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    progress['completed'] += 1
    progress_percentage = (progress['completed'] / total_files) * 100
    print(f"Processed {file_name} \n({progress_percentage:.2f}% complete)")

    return file_name, True

def process_files_in_parallel(input_dir, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.endswith('.mp3')]
    total_files = len(files)

    manager = Manager()
    progress = manager.dict()
    progress['completed'] = 0

    file_info_list = [(file_name, input_dir, output_dir, progress, total_files) for file_name in files]

    with Pool(cpu_count()) as pool:
        pool.map(create_spectrogram, file_info_list)

    print("All files processed.")

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(current_dir, '..', 'bird_recordings')
    output_dir = os.path.join(current_dir, '..', 'spectrograms')

    process_files_in_parallel(input_dir, output_dir)
