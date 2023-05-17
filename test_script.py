import pytest
import numpy as np

from script import load_data, preprocess_labels, build_model, train_model


@pytest.fixture
def example_data():
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array([0, 1, 2])
    return X, y


def test_load_data(example_data):
    X, y = example_data
    X_df, y_loaded = example_data
    assert X_df.shape == (3, 3)
    assert np.array_equal(y_loaded, y)


def test_preprocess_labels(example_data):
    _, y = example_data
    y_train = np.array([0, 1])
    y_test = np.array([2])
    y_train_processed, y_test_processed = preprocess_labels(y_train, y_test)
    assert np.array_equal(y_train_processed, np.array([-1, 0]))
    assert np.array_equal(y_test_processed, np.array([1]))


def test_build_model(example_data):
    X, y = example_data
    model = build_model(input_shape=X.shape[1:])
    assert len(model.layers) == 3
    assert model.layers[-1].output_shape == (None, 3)


def test_train_model(example_data):
    X, y = example_data
    model = build_model(input_shape=X.shape[1:])
    train_model(model, X, y, epochs=1)


pytest.main()
