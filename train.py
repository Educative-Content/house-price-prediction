# Import necessary libraries

import boto3
import os
import time
from io import BytesIO
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense



s3 = boto3.client('s3')
def fetch_documents_from_s3(bucket_name, file=''):
    response = s3.get_object(Bucket=bucket_name, Key=file)
    file = response['Body'].read()
    data = pd.read_csv(BytesIO(file))
    return data


def splitData(data):
    X = data.iloc[:,1:].values #features
    y = data.iloc[:,0].values #price

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(X_train.shape)
    print(X_test.shape)
    return  X_train, X_test, y_train, y_test 

def scaleData(X_train, X_test, y_train, y_test):
    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train.reshape(-1,1)).flatten()
    y_test = scaler_y.fit_transform(y_test.reshape(-1,1)).flatten()
    return scaler, scaler_y, X_train, X_test, y_train, y_test

def buildModel():
    # Build the ANN model
    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_dim=7))
    model.add(Dense(units=1, activation='linear'))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def main():
    files = ['part1.csv','part2.csv','part3.csv']
    bucket_name = os.environ.get('BUCKET_NAME')
    print("Bucket:",bucket_name)
    array_index = int(os.environ.get('AWS_BATCH_JOB_ARRAY_INDEX', 0))
    file= files[array_index]
    print(f"Job no {array_index} starting!")
    print(f"Training on {file}")


    # Load our dataset
    data = fetch_documents_from_s3(bucket_name, file)
    X_train, X_test, y_train, y_test = splitData(data)
    scaler, scaler_y, X_train, X_test, y_train, y_test = scaleData(X_train, X_test, y_train, y_test)
    model = buildModel()


    # Train the model
    start = time.time()
    model.fit(X_train, y_train, epochs=500, batch_size=32, validation_data=(X_test, y_test))
    end = time.time()

    # Evaluate the model on the test set
    loss = model.evaluate(X_test, y_test)
    print(f'Final Loss: {loss}')
    print(f'Time taken: {end - start}s')

    actual_val = scaler_y.inverse_transform(y_test[0].reshape(-1,1))[0][0]
    predictions = model.predict(X_test)
    predicted_val = scaler_y.inverse_transform(predictions[0].reshape(-1,1))[0][0]
    # predicted_val = predictions
    print(f'Actual: {actual_val}, Predicted: {predicted_val}')

    file_name=f'weights{array_index}.npy'

    model_weights = model.get_weights()
    np.save(file_name,np.array(model_weights, dtype=object))
    s3.upload_file(file_name,bucket_name,file_name)
    
    

main()
print('Model has been trained and the weights have been uploaded to the s3 bucket')