import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations (optional)

import numpy as np
import random
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import config

# Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
bird_recordings_dir = os.path.join(current_dir, '../bird_recordings')
curr_model_path = os.path.join(current_dir, config.TEST_MODEL)
label_encoder_path = os.path.join(current_dir, '../model/label_encoder.joblib')

# Load the model
model = load_model(curr_model_path)

# Parameters for MFCC extraction
n_mfcc = 13
max_len = 500  # Ensure this matches the padding length used in training

# Load and preprocess the audio file
def preprocess_audio(file_path, n_mfcc=13, max_len=500):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    if mfccs.shape[1] < max_len:
        pad_width = max_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_len]
    
    return mfccs

# Function to extract the common name from the file name
def extract_common_name(file_name):
    return '_'.join(file_name.split('_')[:-1])

# Get unique species recordings
def get_unique_species_recordings(bird_recordings_dir, num_samples=5):
    files = [file for file in os.listdir(bird_recordings_dir) if file.endswith('.mp3')]
    species_to_files = {}
    
    for file in files:
        species = extract_common_name(file)
        if species not in species_to_files:
            species_to_files[species] = []
        species_to_files[species].append(file)
    
    selected_species = random.sample(list(species_to_files.keys()), min(num_samples, len(species_to_files)))
    selected_files = [random.choice(species_to_files[species]) for species in selected_species]
    
    return selected_files

# Load the label encoder
le = joblib.load(label_encoder_path)

# Get 5 random bird recordings of unique species
selected_files = get_unique_species_recordings(bird_recordings_dir, 5)

# Store results
results = []

# Test each recording with the AI model
for file in selected_files:
    file_path = os.path.join(bird_recordings_dir, file)
    expected_species = extract_common_name(file)
    
    # Preprocess the bird sound
    mfccs = preprocess_audio(file_path)
    mfccs = mfccs[..., np.newaxis]  # Add channel dimension
    mfccs = np.expand_dims(mfccs, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(mfccs)
    predicted_label = np.argmax(predictions, axis=1)
    certainty = np.max(predictions) * 100

    # Map the predicted label to bird species
    predicted_species = le.inverse_transform(predicted_label)[0]

    # Store result
    results.append((expected_species, predicted_species, certainty))

# Print results
correct_predictions = 0
for expected_species, predicted_species, certainty in results:
    print(f'Expected Species: {expected_species}')
    print(f'Predicted Species: {predicted_species}')
    print(f'Certainty: {certainty:.2f}%\n')
    if expected_species == predicted_species:
        correct_predictions += 1

# Print summary
print(f'Correct Predictions: {correct_predictions}/5')
