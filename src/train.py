import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, Conv2D, MaxPooling2D, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import joblib
import librosa
import config
# import audiomentations as am  # Uncomment this if you have the audiomentations library

current_dir = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(current_dir, '../processed_mfccs')
label_encoder_path = os.path.join(current_dir, '../model/label_encoder.joblib')
model_best_path = os.path.join(current_dir, '../model/bird_sound_model_best.keras')
model_final_path = os.path.join(current_dir, '../model/bird_sound_model_final.keras')

# Load the data
def load_data(data_folder, max_allowed_length=500):
    X = []
    y = []
    files = [file for file in os.listdir(data_folder) if file.endswith('.npy')]
    total_files = len(files)

    for i, file in enumerate(files):
        mfccs = np.load(os.path.join(data_folder, file))
        if mfccs.shape[1] <= max_allowed_length:
            X.append(mfccs)
            label = '_'.join(file.split('_')[:-1])  # Extract the label correctly
            y.append(label)
        
            # Display progress
            progress_percentage = (i + 1) / total_files * 100
            print(f'Loading {label}\n({progress_percentage:.2f}% complete)')
        
    print()  # Move to the next line after completing the loop
    return X, y

X, y = load_data(data_folder)

# Cap the maximum length to a reasonable value
max_len = 500  # For example, you can set it to 500

# Pad or truncate the MFCC arrays to the same length manually
def pad_or_truncate(mfccs, max_len):
    if mfccs.shape[1] < max_len:
        pad_width = max_len - mfccs.shape[1]
        return np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
    else:
        return mfccs[:, :max_len]

X_padded = [pad_or_truncate(mfcc, max_len) for mfcc in X]
X = np.array(X_padded, dtype='float32')

# Preprocess the data
X = X[..., np.newaxis]  # Add a new axis for the channel dimension
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Save the label encoder
os.makedirs(os.path.dirname(label_encoder_path), exist_ok=True)
joblib.dump(le, label_encoder_path)

# Data augmentation function
def augment_data(X, y, data_folder, max_len=500):
    aug_X = []
    aug_y = []
    # augmenter = am.Compose([  # Uncomment this if you have the audiomentations library
    #     am.AddBackgroundNoise(sounds_path=config.BACKGROUND_NOISE_PATH, min_snr_in_db=config.MIN_SNR_IN_DB, max_snr_in_db=config.MAX_SNR_IN_DB, p=config.AUGMENTATION_PROBABILITY),
    #     am.TimeStretch(min_rate=config.TIME_STRETCH_MIN_RATE, max_rate=config.TIME_STRETCH_MAX_RATE, p=config.AUGMENTATION_PROBABILITY),
    #     am.PitchShift(min_semitones=-config.PITCH_SHIFT_SEMITONES, max_semitones=config.PITCH_SHIFT_SEMITONES, p=config.AUGMENTATION_PROBABILITY)
    # ])
    
    for i in range(len(X)):
        mfccs = X[i]
        label_index = np.argmax(y[i])
        label = le.inverse_transform([label_index])[0]  # Get the label string
        
        # Original
        aug_X.append(mfccs)
        aug_y.append(y[i])
        
        # Time shifting
        time_shift = np.roll(mfccs, shift=int(mfccs.shape[1] * 0.1), axis=1)
        aug_X.append(time_shift)
        aug_y.append(y[i])
        
        # Add noise
        noise = np.random.normal(0, 0.01, mfccs.shape)
        noisy_mfccs = mfccs + noise
        aug_X.append(noisy_mfccs)
        aug_y.append(y[i])
        
        # Apply audiomentations (uncomment if you have the library and background noises)
        # y_audio, sr = librosa.load(os.path.join(data_folder, f"{label}.mp3"), sr=None)
        # y_augmented = augmenter(samples=y_audio, sample_rate=sr)
        # mfccs_augmented = librosa.feature.mfcc(y=y_augmented, sr=sr, n_mfcc=13)
        # mfccs_augmented = pad_or_truncate(mfccs_augmented, max_len)
        # aug_X.append(mfccs_augmented)
        # aug_y.append(y[i])

    return np.array(aug_X), np.array(aug_y)

X_aug, y_aug = augment_data(X, y_categorical, data_folder, max_len)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_aug, y_aug, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)

# Create the model
input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])

model = Sequential([
    Input(shape=input_shape),
    Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.005)),
    MaxPooling2D(pool_size=(2, 2), padding='same'),
    BatchNormalization(),
    Dropout(0.3),

    Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.005)),
    MaxPooling2D(pool_size=(2, 2), padding='same'),
    BatchNormalization(),
    Dropout(0.3),

    Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.005)),
    MaxPooling2D(pool_size=(2, 2), padding='same'),
    BatchNormalization(),
    Dropout(0.3),

    Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.005)),
    MaxPooling2D(pool_size=(2, 2), padding='same'),
    BatchNormalization(),
    Dropout(0.3),

    Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.005)),
    MaxPooling2D(pool_size=(2, 2), padding='same'),
    BatchNormalization(),
    Dropout(0.3),

    Flatten(),
    Dense(512, activation='relu', kernel_regularizer=l2(0.005)),
    Dropout(0.4),
    BatchNormalization(),

    Dense(len(le.classes_), activation='softmax')
])

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=config.PATIENCE_EARLY_STOPPING, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(model_best_path, save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=config.FACTOR_REDUCE_LR, patience=config.PATIENCE_REDUCE_LR, min_lr=config.MIN_LR)

# Train the model
model.fit(X_train, y_train, batch_size=config.BATCH_SIZE, epochs=config.EPOCHS, verbose=1, validation_data=(X_test, y_test),
          callbacks=[early_stopping, model_checkpoint, reduce_lr])

# Evaluate the model
score = model.evaluate(X_test, y_test, verbose=0)
print(f'Test loss: {score[0]}')
print(f'Test accuracy: {score[1]}')

# Save the final model
model.save(model_final_path)
