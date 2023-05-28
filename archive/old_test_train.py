import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import InputLayer, Dense
import mlflow
import random
import os
import tensorflow as tf
from train import *

def test_get_data():
    X, y = get_data()
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape[0] == y.shape[0]


def test_process_data():
    X, y = get_data()
    X_train, y_train, X_test, y_test = process_data(X, y)
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_test, pd.Series)
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]


def test_create_model():
    input_shape = 10
    model = create_model(input_shape)
    assert isinstance(model, Sequential)


def test_config_mlflow():
    config_mlflow()
    assert isinstance(os.environ['MLFLOW_TRACKING_USERNAME'], str)
    assert isinstance(os.environ['MLFLOW_TRACKING_PASSWORD'], str)
    assert isinstance(mlflow.get_tracking_uri(), str)


def test_train_model():
    mlflow.set_tracking_uri('')
    input_shape = 4
    X, y = get_data()
    X_train, y_train, X_test, y_test = process_data(X, y)
    model = create_model(input_shape)
    train_model(model, X_train, y_train, 1)
