import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, average_precision_score
from scipy.stats import pearsonr, spearmanr

def xgb_results_regression(x_train, x_test, y_train, y_test):


    # Set the parameters for XGBoost
    xgb_params = {
        'n_estimators': 100,
        'n_jobs': -1,
        'random_state': 42, 
    }

    # Initialize XGBoost regressor with custom parameters
    model = xgb.XGBRegressor(**xgb_params)

    # Train the model
    model.fit(x_train, y_train)

    # Make predictions on training set
    train_predictions = model.predict(x_train)

    # Make predictions on testing set
    test_predictions = model.predict(x_test)

    # Calculate evaluation metrics for training set
    train_metrics = {
        'MSE': mean_squared_error(y_train, train_predictions),
        'R^2 Score': r2_score(y_train, train_predictions),
        'Pearson': pearsonr(y_train, train_predictions)[0],
        'Spearman': spearmanr(y_train, train_predictions)[0],
    }

    # Calculate evaluation metrics for testing set
    test_metrics = {
        'MSE': mean_squared_error(y_test, test_predictions),
        'R^2 Score': r2_score(y_test, test_predictions),
        'Pearson': pearsonr(y_test, test_predictions)[0],
        'Spearman': spearmanr(y_test, test_predictions)[0],
    }

    # Return the results as a dictionary
    results = {
        'Training Metrics': train_metrics,
        'Testing Metrics': test_metrics
    }

    return results




    