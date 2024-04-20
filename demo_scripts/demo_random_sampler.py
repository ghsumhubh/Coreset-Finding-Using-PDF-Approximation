import numpy as np
def sample_data(x_train, y_train, sample_size):
    indices = np.random.choice(x_train.shape[0], sample_size, replace=False)
    smaller_x_train = x_train[indices]
    smaller_y_train = y_train[indices]

    return smaller_x_train, smaller_y_train