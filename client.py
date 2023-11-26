import numpy as np
import pandas as pd
import os
import sys
import flwr
import argparse
import ipaddress

from typing import Dict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import RobustScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, InputLayer

class Client(flwr.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(X_train, y_train, epochs=25, validation_data=(X_test, y_test), batch_size=32, verbose=0)
        return model.get_weights(), len(X_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(X_test, y_test)
        return loss, len(X_test), {"accuracy": round(accuracy, 4), "loss": loss}

if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description='Federated Learning Classification Demo')
    parser.add_argument("--address", help="IP Address", default="127.0.0.1")
    parser.add_argument("--port", help="Serving port", default=8000, type=int)
    parser.add_argument("--train_path", help="Training dataset path", default="/mnt/hdd/Datasets/wefe-default_data/water/water_subset_1.csv")
    parser.add_argument("--test_path", help="Test dataset path", default="/mnt/hdd/Datasets/wefe-default_data/water/water_subset_1.csv")
    parser.add_argument("--target", help="Target column name", default="y")
    args = parser.parse_args()

    try:
        ipaddress.ip_address(args.address)
    except ValueError:
        sys.exit(f"Wrong IP address: {args.address}")

    if args.port < 0 or args.port > 65535:
        sys.exit(f"Wrong port number: {args.port}")

    if not os.path.exists(args.train_path):
        sys.exit(f"Train file not found: {args.train_path}")

    if not os.path.exists(args.test_path):
        sys.exit(f"Test file not found in {args.test_path}")

    df_train = pd.read_csv(args.train_path)
    df_test = pd.read_csv(args.test_path)

    X_train = df_train.drop([args.target, 'index'], axis=1).to_numpy()
    X_test = df_test.drop([args.target, 'index'], axis=1).to_numpy()

    y_train = df_train[args.target].to_numpy()
    y_test = df_test[args.target].to_numpy()

    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = Sequential([
        InputLayer(input_shape=(X_train.shape[1],)),
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    flwr.client.start_numpy_client(
        server_address=f"{args.address}:{args.port}",
        client=Client()
    )