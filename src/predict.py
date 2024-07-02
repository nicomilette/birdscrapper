def predict_bird_sound(file_path):
    mfccs = np.load(file_path)
    mfccs = mfccs[np.newaxis, ..., np.newaxis]  # Add batch and channel dimensions
    predictions = model.predict(mfccs)
    predicted_label = le.inverse_transform(np.argmax(predictions, axis=1))
    certainty = np.max(predictions)
    return predicted_label[0], certainty

# Example prediction
file_path = '../processed_mfccs/example_bird.npy'
predicted_bird, certainty = predict_bird_sound(file_path)
print(f'Predicted Bird: {predicted_bird}, Certainty: {certainty}')