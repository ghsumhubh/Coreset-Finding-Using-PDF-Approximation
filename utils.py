import numpy as np
import pandas as pd

def convert_to_numpy_dataset(x_train, x_test, y_train, y_test):
    # convert to numpy arrays
    x_train = x_train.to_numpy()
    x_test = x_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    # make sure the y is one dimensional
    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)
    print("Number of features: ", x_train.shape[1])
    print("Number of training examples: ", x_train.shape[0])
    print("Number of test examples: ", x_test.shape[0])
    return x_train, x_test, y_train, y_test

def make_categorical_into_onehot(X, y, columns_to_onehot):
    X = pd.get_dummies(X, columns=columns_to_onehot)
    return X, y


def sample_data(x_train, y_train, sample_rate, seed):
    np.random.seed(seed)
    sample_size = int(x_train.shape[0] * sample_rate)
    indices = np.random.choice(x_train.shape[0], sample_size, replace=False)
    smaller_x_train = x_train[indices]
    smaller_y_train = y_train[indices]

    return smaller_x_train, smaller_y_train