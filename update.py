# Import necessary libraries
import boto3
import os
from io import BytesIO
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense



def getModel(key):
    s3.download_file(bucket_name, key, f"./{key}")
    model = load_model(f"./{key}")
    return model



bucket_name = os.environ.get('BUCKET_NAME')
weights_key = ["weights0.npy","weights1.npy","weights2.npy"]

s3 = boto3.client('s3')
new_weights = []
for index, weight in enumerate(weights_key):
    response = s3.get_object(Bucket=bucket_name, Key=f"weights/{weight}")
    file = BytesIO(response["Body"].read())
    array = np.load(file, allow_pickle=True)
    new_weights.append(array)

new_weights = np.array(new_weights)
new_weights = np.mean(new_weights, axis=0)
print("Mean weights have been calculated!")

# Loading the model by downloading it from the s3 Bucket
model = getModel("model.h5")

 # Build the ANN model
model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=7))
model.add(Dense(units=1, activation='linear'))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

model.set_weights(new_weights)
print("Model weights have been set!")

# Saving the model
model.save('model.h5')    

s3.upload_file("model.h5",bucket_name, "model.h5")
print(f'Uploaded model to {bucket_name}')
