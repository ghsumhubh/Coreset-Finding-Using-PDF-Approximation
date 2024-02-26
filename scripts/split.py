import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

TEST_SIZE = 0.2

def split_data(x, y):
    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=42)
    
    return x_train, x_test, y_train, y_test

