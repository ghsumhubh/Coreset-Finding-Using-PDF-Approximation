

import numpy as np
from scipy import stats

def remove_outliers(x_train, y_train, outlier_threshold=3):
    # Calculate Z-scores of features in x_train
    z_scores = np.abs(stats.zscore(x_train, axis=0))
    
    # Find indices where any feature has a Z-score greater than the threshold
    # np.any(axis=1) ensures that we look across all features for each observation
    indices = np.where(np.any(z_scores > outlier_threshold, axis=1))
    
    # Remove the outliers based on the identified indices
    x_train_clean = np.delete(x_train, indices, axis=0)
    y_train_clean = np.delete(y_train, indices, axis=0)
    
    return x_train_clean, y_train_clean