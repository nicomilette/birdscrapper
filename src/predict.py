import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib  # Add joblib for loading the label encoder

# Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
bird_path = os.path.join(current_dir, '../bird_recordings/Abert_s Towhee_111838.mp3')
curr_model_path = os.path.join(current_dir, '../model/bird_sound_model_best.keras')
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

# Preprocess the bird sound
mfccs = preprocess_audio(bird_path)
mfccs = mfccs[..., np.newaxis]  # Add channel dimension

# Add batch dimension
mfccs = np.expand_dims(mfccs, axis=0)

# Make prediction
predictions = model.predict(mfccs)
predicted_label = np.argmax(predictions, axis=1)

# Load the label encoder and map the predicted label to bird species
le = joblib.load(label_encoder_path)
bird_species = le.inverse_transform(predicted_label)

print(f'Predicted Bird Species: {bird_species[0]}')
