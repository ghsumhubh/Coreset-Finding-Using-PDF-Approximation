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

    valid_columns = list(set(columns_to_onehot).intersection(set(X.columns)))
    #print("Categorical columns: ", columns_to_onehot)
    #print("Valid columns: ", valid_columns)
    X = pd.get_dummies(X, columns=valid_columns)
    return X, y


def sample_data(x_train, y_train, sample_size):
    indices = np.random.choice(x_train.shape[0], sample_size, replace=False)
    smaller_x_train = x_train[indices]
    smaller_y_train = y_train[indices]

    return smaller_x_train, smaller_y_train





def get_baseline_guesses(y):
    mean_guess = np.mean(y)
    median_guess = np.median(y)
    guesses = {
        'mean': mean_guess,
        'median': median_guess
    }
    return guesses

def get_baseline_results(y, guesses):
    # for each guess calculate the mse and r2
    results = {}
    for key, value in guesses.items():
        mse = np.mean((y - value) ** 2)
        r2 = 1 - (np.sum((y - value) ** 2) / np.sum((y - np.mean(y)) ** 2))
        results[key] = {
            'MSE': mse,
            'R^2': r2
        }
    return results