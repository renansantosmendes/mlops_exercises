import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, InputLayer

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import os
import git
import random
import numpy as np


def reset_seeds():
    os.environ['PYTHONHASHSEED'] = str(42)
    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)


def load_data():
    base_data_url = 'https://github.com/renansantosmendes/lectures-cdas-2023.git'
    if not os.path.isdir('lectures-cdas-2023'):
        git.Repo.clone_from(base_data_url, 'lectures-cdas-2023')
    data = pd.read_csv(os.path.join('lectures-cdas-2023', 'fetal_health_reduced.csv'))
    X = data.drop(["fetal_health"], axis=1)
    y = data["fetal_health"]

    columns_names = list(X.columns)
    scaler = preprocessing.StandardScaler()
    X_df = scaler.fit_transform(X)
    X_df = pd.DataFrame(X_df, columns=columns_names)

    return X_df, y


def preprocess_labels(y_train, y_test):
    y_train = y_train - 1
    y_test = y_test - 1
    return y_train, y_test


def build_model(input_shape):
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def train_model(model, X_train, y_train, epochs):
    model.fit(X_train, y_train, epochs=epochs, validation_split=0.2, verbose=3)


if __name__ == "__main__":
    reset_seeds()
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    y_train, y_test = preprocess_labels(y_train, y_test)
    model = build_model(input_shape=X_train.shape[1:])
    train_model(model, X_train, y_train, epochs=50)