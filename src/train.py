import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense
import pickle

# Load preprocessed data
with open("data/preprocessed_data.pkl", "rb") as f:
    X = pickle.load(f)

# Load pre-trained model
pretrained_model = load_model("models/pretrained_lstm.h5")

# Freeze initial layers
for layer in pretrained_model.layers[:-2]:  
    layer.trainable = False  

# Add new dense layers
new_model = Sequential([
    pretrained_model,
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Predict failure probability
])

# Compile model
new_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
history = new_model.fit(X, epochs=20, batch_size=32, validation_split=0.2)

# Save fine-tuned model
new_model.save("models/fine_tuned_lstm.h5")
print("Fine-tuned model saved.")
