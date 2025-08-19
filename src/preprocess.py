import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler

# Load dataset
df = pd.read_csv("data/air_filter_data.csv")

# Handle missing values
df.fillna(method='ffill', inplace=True)

# Select relevant features
sensor_cols = ["airflow", "pressure", "temperature", "filter_age"]
df_selected = df[sensor_cols]

# Normalize data
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_selected), columns=sensor_cols)

# Save preprocessed data
with open("data/preprocessed_data.pkl", "wb") as f:
    pickle.dump(df_scaled, f)

print("Data preprocessing complete. Saved as preprocessed_data.pkl")
