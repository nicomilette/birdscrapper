import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from skimage import io
from tensorflow.keras.models import load_model


current_dir = os.path.dirname(os.path.abspath(__file__))
bird_path = os.path.join(current_dir, '../bird_recordings/Song Sparrow_17143')
curr_model_path = os.path.join(current_dir, '../bird_sound_model_final1.keras')
model = load_model(curr_model_path)


def convert_mp3_to_mfcc(file_path):
    try:
        # Load the MP3 file
        y, sr = librosa.load(file_path, sr=None)
        
        # Reduce noise
        y_reduced = nr.reduce_noise(y=y, sr=sr)
        
        # Normalize audio
        y_normalized = librosa.util.normalize(y_reduced)
        
        # Create a mel spectrogram
        S = librosa.feature.melspectrogram(y=y_normalized, sr=sr)
        S_DB = librosa.power_to_db(S, ref=np.max)
        
        # Plot and save the spectrogram
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel')
        plt.axis('off')
        plt.tight_layout(pad=0)
        spectrogram_path = file_path.replace('.mp3', '.png')
        plt.savefig(spectrogram_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # Load the spectrogram
        spectrogram = io.imread(spectrogram_path, as_gray=True)
        
        # Convert the spectrogram image back to audio signal and extract MFCCs
        y_from_spec = librosa.feature.inverse.mel_to_audio(spectrogram, sr=sr)
        mfccs = librosa.feature.mfcc(y=y_from_spec, sr=sr, n_mfcc=13)
        
        return mfccs
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def predict_bird_sound(file_path):
    mfccs = convert_mp3_to_mfcc(file_path)
    if mfccs is None:
        return "Error", 0.0

    mfccs = mfccs[np.newaxis, ..., np.newaxis]  # Add batch and channel dimensions
    predictions = model.predict(mfccs)
    predicted_label = le.inverse_transform(np.argmax(predictions, axis=1))
    certainty = np.max(predictions)
    return predicted_label[0], certainty

# Example prediction
predicted_bird, certainty = predict_bird_sound(bird_path)
print(f'Predicted Bird: {predicted_bird}, Certainty: {certainty}')
