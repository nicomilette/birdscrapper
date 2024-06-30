import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Ensure reproducibility
tf.random.set_seed(42)

# Step 2: Create a Dataset
def create_dataset(data_dir, batch_size=32, img_height=180, img_width=180):
    datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)
    train_data = datagen.flow_from_directory(
        data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    val_data = datagen.flow_from_directory(
        data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    return train_data, val_data

# Adjust the path to your dataset
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '..', 'spectrograms')
train_data, val_data = create_dataset(data_dir)

# Step 3: Train the Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(180, 180, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(train_data.class_indices), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_data, validation_data=val_data, epochs=10)

# Step 4: Evaluate the Model
test_loss, test_acc = model.evaluate(val_data, verbose=2)
print(f'\nTest accuracy: {test_acc}')

# Step 5: Save the Model
model.save(os.path.join(current_dir, 'bird_call_classifier.h5'))
print("Model saved to 'bird_call_classifier.h5'")
