import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

current_dir = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(current_dir, '../processed_mfccs')

# Load the data
def load_data(data_folder):
    X = []
    y = []
    files = [file for file in os.listdir(data_folder) if file.endswith('.npy')]
    total_files = len(files)

    for i, file in enumerate(files):
        mfccs = np.load(os.path.join(data_folder, file))
        X.append(mfccs)
        label = '_'.join(file.split('_')[:-1])  # Extract the label correctly
        y.append(label)
        
        # Display progress
        progress_percentage = (i + 1) / total_files * 100
        print(f'Loading {label}\n({progress_percentage:.2f}% complete)')
        
    print()  # Move to the next line after completing the loop
    return np.array(X), np.array(y)

X, y = load_data(data_folder)

# Preprocess the data
X = X[..., np.newaxis]  # Add a new axis for the channel dimension
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Data augmentation function
def augment_data(X, y):
    aug_X = []
    aug_y = []
    for i in range(len(X)):
        mfccs = X[i]
        label = y[i]
        
        # Original
        aug_X.append(mfccs)
        aug_y.append(label)
        
        # Time shifting
        time_shift = np.roll(mfccs, shift=int(mfccs.shape[1] * 0.1), axis=1)
        aug_X.append(time_shift)
        aug_y.append(label)
        
        # Add noise
        noise = np.random.normal(0, 0.01, mfccs.shape)
        noisy_mfccs = mfccs + noise
        aug_X.append(noisy_mfccs)
        aug_y.append(label)
        
        # Pitch shifting can be added similarly if applicable
        
    return np.array(aug_X), np.array(aug_y)

X_aug, y_aug = augment_data(X, y_categorical)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_aug, y_aug, test_size=0.2, random_state=42)

# Create the model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=X_train.shape[1:]),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),
    
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),
    
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),
    
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    BatchNormalization(),
    
    Dense(len(le.classes_), activation='softmax')
])

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('bird_sound_model_best.keras', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6)

# Train the model
model.fit(X_train, y_train, batch_size=64, epochs=100, verbose=1, validation_data=(X_test, y_test),
          callbacks=[early_stopping, model_checkpoint, reduce_lr])

# Evaluate the model
score = model.evaluate(X_test, y_test, verbose=0)
print(f'Test loss: {score[0]}')
print(f'Test accuracy: {score[1]}')

# Save the final model
model.save('bird_sound_model_final.keras')
