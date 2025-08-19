import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

# Load preprocessed test data
with open("data/preprocessed_data.pkl", "rb") as f:
    X_test = pickle.load(f)

# Load trained model
model = load_model("models/fine_tuned_lstm.h5")

# Predict on test data
predictions = model.predict(X_test)

# Display results
for i, pred in enumerate(predictions[:5]):
    print(f"Sample {i+1}: Failure Probability = {pred[0]:.2f}")
