import os
import numpy as np
import librosa
import noisereduce as nr
import requests
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count, Manager
import config

# Define directories
current_dir = os.path.dirname(os.path.abspath(__file__))
recordings_dir = os.path.join(current_dir, '../bird_recordings')
processed_mfccs_dir = os.path.join(current_dir, '../processed_mfccs')

# Helper functions
def clean_filename(name):
    return "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in name)

def load_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        return y, sr
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None, None

def reduce_noise(y, sr):
    try:
        y_reduced = nr.reduce_noise(y=y, sr=sr)
        return y_reduced
    except Exception as e:
        print(f"Error reducing noise: {e}")
        return y

def normalize_audio(y):
    try:
        if not np.all(np.isfinite(y)):
            raise ValueError("Non-finite values detected in audio data")
        y_normalized = librosa.util.normalize(y)
        return y_normalized
    except Exception as e:
        print(f"Error normalizing audio: {e}")
        return y

def extract_mfcc(y, sr, n_mfcc=13):
    try:
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return mfccs
    except Exception as e:
        print(f"Error extracting MFCCs: {e}")
        return None

def verify_mp3(file_info):
    file_name, recordings_dir, progress, total_files, lock = file_info
    file_path = os.path.join(recordings_dir, file_name)

    try:
        y, sr = load_audio(file_path)
        if y is None or sr is None:
            raise ValueError("Error loading audio")
        
        # Check if audio data is valid
        if not np.all(np.isfinite(y)) or len(y) == 0:
            raise ValueError("Invalid audio data")
        
        with lock:
            progress['completed'] += 1
            progress_percentage = (progress['completed'] / total_files) * 100
            print(f"Verified {file_name}.")
            print(f"Processed {progress['completed']}/{total_files} files \n({progress_percentage:.2f}% complete)")

    except Exception as e:
        print(f"Error verifying file {file_name}: {e}")
        redownload_file(file_name)
        
        with lock:
            progress['completed'] += 1
            progress_percentage = (progress['completed'] / total_files) * 100
            print(f"Redownloaded {file_name}.")
            print(f"Processed {progress['completed']}/{total_files} files \n({progress_percentage:.2f}% complete)")

def redownload_file(file_name):
    file_path = os.path.join(recordings_dir, file_name)
    recording_id = file_name.split('_')[-1].replace('.mp3', '')
    common_name = file_name.rsplit('_', 1)[0]

    # Load the filtered dataset
    file_path_csv = os.path.join(current_dir, config.CSV_FILTERED)
    bird_data = pd.read_csv(file_path_csv)
    row = bird_data[bird_data['id'] == int(recording_id)].iloc[0]

    recording_url = row['recording_url']

    try:
        response = requests.get(recording_url, stream=True)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
        else:
            print(f"Failed to redownload {recording_url}")
    except Exception as e:
        print(f"Error redownloading {recording_url}: {e}")

def verify_npy(file_info):
    file_name, recordings_dir, processed_mfccs_dir, progress, total_files, lock = file_info
    file_path = os.path.join(recordings_dir, file_name)
    output_path = os.path.join(processed_mfccs_dir, file_name.replace('.mp3', '.npy'))

    try:
        if not os.path.exists(output_path):
            raise ValueError("Processed MFCC file does not exist")

        mfccs = np.load(output_path)
        if not np.all(np.isfinite(mfccs)) or mfccs is None or len(mfccs) == 0:
            raise ValueError("Invalid MFCC data")

        with lock:
            progress['completed'] += 1
            progress_percentage = (progress['completed'] / total_files) * 100
            print(f"Verified {file_name}.")
            print(f"Processed {progress['completed']}/{total_files} files \n({progress_percentage:.2f}% complete)")

    except Exception as e:
        print(f"Error verifying file {file_name}: {e}")
        preprocess_audio((file_name, recordings_dir, processed_mfccs_dir, progress, total_files, lock))
        
        with lock:
            progress['completed'] += 1
            progress_percentage = (progress['completed'] / total_files) * 100
            print(f"Reprocessed {file_name}.")
            print(f"Processed {progress['completed']}/{total_files} files \n({progress_percentage:.2f}% complete)")

def preprocess_audio(file_info):
    file_name, input_folder, output_folder, progress, total_files, lock = file_info
    file_path = os.path.join(input_folder, file_name)
    output_path = os.path.join(output_folder, file_name.replace('.mp3', '.npy'))

    if os.path.exists(output_path):
        with lock:
            progress['completed'] += 1
            progress_percentage = (progress['completed'] / total_files) * 100
            print(f"File {file_name} already processed. Skipping conversion.")
            print(f"Processed {progress['completed']}/{total_files} files \n({progress_percentage:.2f}% complete)")
        return

    y, sr = load_audio(file_path)
    if y is None or sr is None:
        with lock:
            progress['completed'] += 1
            progress_percentage = (progress['completed'] / total_files) * 100
            print(f"File {file_name} could not be loaded. Skipping.")
            print(f"Processed {progress['completed']}/{total_files} files \n({progress_percentage:.2f}% complete)")
        return

    y_reduced = reduce_noise(y, sr)
    y_normalized = normalize_audio(y_reduced)
    if y_normalized is None:
        with lock:
            progress['completed'] += 1
            progress_percentage = (progress['completed'] / total_files) * 100
            print(f"Normalization failed for file {file_name}. Skipping.")
            print(f"Processed {progress['completed']}/{total_files} files \n({progress_percentage:.2f}% complete)")
        return

    mfccs = extract_mfcc(y_normalized, sr)
    if mfccs is None:
        with lock:
            progress['completed'] += 1
            progress_percentage = (progress['completed'] / total_files) * 100
            print(f"MFCC extraction failed for file {file_name}. Skipping.")
            print(f"Processed {progress['completed']}/{total_files} files \n({progress_percentage:.2f}% complete)")
        return

    np.save(output_path, mfccs)

    with lock:
        progress['completed'] += 1
        progress_percentage = (progress['completed'] / total_files) * 100
        print(f"Processed {progress['completed']}/{total_files} files \n({progress_percentage:.2f}% complete)")

def verify_and_download_recordings_sequential():
    file_path = os.path.join(current_dir, config.CSV_FILTERED)
    bird_data = pd.read_csv(file_path)
    total_recordings = bird_data.shape[0]
    downloaded_recordings = 0

    for index, row in bird_data.iterrows():
        recording_url = row['recording_url']
        recording_id = row['id']
        common_name = row['common_name']
        file_name = f"{clean_filename(common_name)}_{recording_id}.mp3"
        output_path = os.path.join(recordings_dir, file_name)
        
        if os.path.exists(output_path):
            try:
                y, sr = load_audio(output_path)
                if y is None or sr is None or not np.all(np.isfinite(y)) or len(y) == 0:
                    raise ValueError("Invalid audio data")
                print(f"File {file_name} verified successfully.")
            except Exception as e:
                print(f"File {file_name} is invalid: {e}. Redownloading...")
                redownload_file(file_name)
        else:
            download_recording(recording_url, output_path)
        
        downloaded_recordings += 1
        progress_percentage = (downloaded_recordings / total_recordings) * 100
        print(f"Processed {file_name} \n({progress_percentage:.2f}% complete)")

    print("All recordings verified and downloaded.")

def verify_and_preprocess_files(input_folder, output_folder, parallel=True):
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    files = [f for f in os.listdir(input_folder) if f.endswith('.mp3')]
    total_files = len(files)

    if parallel:
        manager = Manager()
        progress = manager.dict()
        progress['completed'] = len([f for f in os.listdir(output_folder) if f.endswith('.npy')])
        lock = manager.Lock()

        file_info_list = [(file_name, input_folder, output_folder, progress, total_files, lock) for file_name in files]

        with Pool(cpu_count()) as pool:
            pool.map(verify_npy, file_info_list)
    else:
        progress = {'completed': len([f for f in os.listdir(output_folder) if f.endswith('.npy')])}
        lock = None  # No need for a lock in sequential processing

        file_info_list = [(file_name, input_folder, output_folder, progress, total_files, lock) for file_name in files]

        for file_info in file_info_list:
            verify_npy(file_info)

    print("All files processed.")

def verify_files(file_type, parallel=True):
    if file_type == 'mp3':
        print("Verifying MP3 files...")
        if parallel:
            verify_and_download_recordings_sequential()
        else:
            verify_and_download_recordings_sequential()
    elif file_type == 'npy':
        print("Verifying NPY files...")
        verify_and_preprocess_files(recordings_dir, processed_mfccs_dir, parallel)
    else:
        print("Invalid file type. Please choose 'mp3' or 'npy'.")

if __name__ == '__main__':
    
    choice = input("1. Verify MP3 files\n2. Verify NPY files\nEnter your choice (1-2): ")
    if choice == '1':
        sub_choice = input("1. Parallel processing\n2. Sequential processing\nEnter your choice (1-2): ")
        if sub_choice == '1':
            verify_files('mp3', parallel=True)
        elif sub_choice == '2':
            verify_files('mp3', parallel=False)
        else:
            print("Invalid choice. Please enter 1 or 2.")
    elif choice == '2':
        sub_choice = input("1. Parallel processing\n2. Sequential processing\nEnter your choice (1-2): ")
        if sub_choice == '1':
            verify_files('npy', parallel=True)
        elif sub_choice == '2':
            verify_files('npy', parallel=False)
        else:
            print("Invalid choice. Please enter 1 or 2.")
    else:
        print("Invalid choice. Please enter 1 or 2.")
