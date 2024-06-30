import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import librosa.display
from multiprocessing import Pool, cpu_count, Manager

def create_spectrogram(file_info):
    file_name, input_dir, output_dir, progress, total_files, lock = file_info
    input_path = os.path.join(input_dir, file_name)
    output_path = os.path.join(output_dir, file_name.replace('.mp3', '.png'))

    if os.path.exists(output_path):
        if lock:
            with lock:
                progress['completed'] += 1
                progress_percentage = (progress['completed'] / total_files) * 100
                print(f"File {file_name} already processed. Skipping conversion.")
                print(f"Processed {progress['completed']}/{total_files} files ({progress_percentage:.2f}% complete)")
        else:
            progress['completed'] += 1
            progress_percentage = (progress['completed'] / total_files) * 100
            print(f"File {file_name} already processed. Skipping conversion.")
            print(f"Processed {progress['completed']}/{total_files} files ({progress_percentage:.2f}% complete)")
        return file_name, False

    if not os.path.exists(input_path):
        if lock:
            with lock:
                progress['completed'] += 1
                progress_percentage = (progress['completed'] / total_files) * 100
                print(f"File {file_name} does not exist. Skipping.")
                print(f"Processed {progress['completed']}/{total_files} files ({progress_percentage:.2f}% complete)")
        else:
            progress['completed'] += 1
            progress_percentage = (progress['completed'] / total_files) * 100
            print(f"File {file_name} does not exist. Skipping.")
            print(f"Processed {progress['completed']}/{total_files} files ({progress_percentage:.2f}% complete)")
        return file_name, False

    try:
        y, sr = librosa.load(input_path, sr=None)
        plt.figure(figsize=(10, 4))
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_DB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    except Exception as e:
        if lock:
            with lock:
                progress['completed'] += 1
                progress_percentage = (progress['completed'] / total_files) * 100
                print(f"Error processing file {file_name}: {e}")
                print(f"Processed {progress['completed']}/{total_files} files \n({progress_percentage:.2f}% complete)")
        else:
            progress['completed'] += 1
            progress_percentage = (progress['completed'] / total_files) * 100
            print(f"Error processing file {file_name}: {e}")
            print(f"Processed {progress['completed']}/{total_files} files \n({progress_percentage:.2f}% complete)")
        return file_name, False

    if lock:
        with lock:
            progress['completed'] += 1
            progress_percentage = (progress['completed'] / total_files) * 100
            print(f"Processed {progress['completed']}/{total_files} files \n({progress_percentage:.2f}% complete)")
    else:
        progress['completed'] += 1
        progress_percentage = (progress['completed'] / total_files) * 100
        print(f"Processed {progress['completed']}/{total_files} files \n({progress_percentage:.2f}% complete)")

    return file_name, True

def process_files_in_parallel(input_dir, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.endswith('.mp3')]
    total_files = len(files)

    manager = Manager()
    progress = manager.dict()
    progress['completed'] = len([f for f in os.listdir(output_dir) if f.endswith('.png')])
    lock = manager.Lock()

    file_info_list = [(file_name, input_dir, output_dir, progress, total_files, lock) for file_name in files]

    with Pool(cpu_count()) as pool:
        pool.map(create_spectrogram, file_info_list)

    print("All files processed.")

def process_files_sequentially(input_dir, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.endswith('.mp3')]
    total_files = len(files)

    progress = {'completed': len([f for f in os.listdir(output_dir) if f.endswith('.png')])}
    lock = None  # No need for a lock in sequential processing

    file_info_list = [(file_name, input_dir, output_dir, progress, total_files, lock) for file_name in files]

    for file_info in file_info_list:
        create_spectrogram(file_info)

    print("All files processed.")

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(current_dir, '..', 'bird_recordings')
    output_dir = os.path.join(current_dir, '..', 'spectrograms')

    choice = input("1. Parallel computation\n2. Sequential processing\n")

    if choice == '1':
        process_files_in_parallel(input_dir, output_dir)
    elif choice == '2':
        process_files_sequentially(input_dir, output_dir)
    else:
        print("Invalid choice. Please enter 1 or 2.")
