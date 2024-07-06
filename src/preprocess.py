import os
import numpy as np
import librosa
import noisereduce as nr
from pathlib import Path
from multiprocessing import Pool, cpu_count, Manager

def load_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        return y, sr
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None, None

def reduce_noise(y, sr):
    y_reduced = nr.reduce_noise(y=y, sr=sr)
    return y_reduced

def normalize_audio(y):
    y_normalized = librosa.util.normalize(y)
    return y_normalized

def extract_mfcc(y, sr, n_mfcc=13):
    try:
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return mfccs
    except Exception as e:
        print(f"Error extracting MFCCs: {e}")
        return None

def preprocess_audio(file_info):
    file_name, input_folder, output_folder, progress, total_files, lock = file_info
    file_path = os.path.join(input_folder, file_name)
    output_path = os.path.join(output_folder, file_name.replace('.mp3', '.npy'))

    if os.path.exists(output_path):
        with lock:
            progress_percentage = (progress['completed'] / total_files) * 100
            print(f"File {file_name} already processed. Skipping conversion.")
            print(f"Processed {progress['completed']}/{total_files} files \n({progress_percentage:.2f}% complete)")
        return

    y, sr = load_audio(file_path)
    if y is None or sr is None:
        with lock:
            progress_percentage = (progress['completed'] / total_files) * 100
            print(f"File {file_name} could not be loaded. Skipping.")
            print(f"Processed {progress['completed']}/{total_files} files \n({progress_percentage:.2f}% complete)")
        return

    y_reduced = reduce_noise(y, sr)
    y_normalized = normalize_audio(y_reduced)
    mfccs = extract_mfcc(y_normalized, sr)
    if mfccs is None:
        with lock:
            progress_percentage = (progress['completed'] / total_files) * 100
            print(f"MFCC extraction failed for file {file_name}. Skipping.")
            print(f"Processed {progress['completed']}/{total_files} files \n({progress_percentage:.2f}% complete)")
        return

    np.save(output_path, mfccs)

    with lock:
        progress['completed'] += 1
        progress_percentage = (progress['completed'] / total_files) * 100
        print(f"Processed {progress['completed']}/{total_files} files \n({progress_percentage:.2f}% complete)")

def process_files_in_parallel(input_folder, output_folder):
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    files = [f for f in os.listdir(input_folder) if f.endswith('.mp3')]
    total_files = len(files)

    manager = Manager()
    progress = manager.dict()
    progress['completed'] = len([f for f in os.listdir(output_folder) if f.endswith('.npy')])
    lock = manager.Lock()

    file_info_list = [(file_name, input_folder, output_folder, progress, total_files, lock) for file_name in files]

    with Pool(cpu_count()) as pool:
        pool.map(preprocess_audio, file_info_list)

    print("All files processed.")

def process_files_sequentially(input_folder, output_folder):
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    files = [f for f in os.listdir(input_folder) if f.endswith('.mp3')]
    total_files = len(files)

    progress = {'completed': len([f for f in os.listdir(output_folder) if f.endswith('.npy')])}
    lock = None  # No need for a lock in sequential processing

    file_info_list = [(file_name, input_folder, output_folder, progress, total_files, lock) for file_name in files]

    for file_info in file_info_list:
        preprocess_audio(file_info)

    print("All files processed.")

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(script_dir, '../bird_recordings')
    output_folder = os.path.join(script_dir, '../processed_mfccs')

    choice = input("1. Parallel computation\n2. Sequential processing\n")

    if choice == '1':
        process_files_in_parallel(input_folder, output_folder)
    elif choice == '2':
        process_files_sequentially(input_folder, output_folder)
    else:
        print("Invalid choice. Please enter 1 or 2.")
