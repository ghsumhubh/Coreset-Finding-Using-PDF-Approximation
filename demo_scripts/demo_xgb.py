import xgboost as xgb
import shap
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

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

    return mean_squared_error(y_test, test_predictions)



def get_shap_values_after_xgb(x_train, y_train):
    xgb_params = {
        'n_estimators': 100,
        'n_jobs': -1,
        'random_state': 42, 
    }


    # Initialize XGBoost regressor with custom parameters
    model = xgb.XGBRegressor(**xgb_params)

    # Train the model
    model.fit(x_train, y_train)

    # Create a SHAP explainer object using the model
    explainer = shap.Explainer(model)

    # Calculate SHAP values for the training set
    shap_values = explainer(x_train)

    # Aggregate the absolute SHAP values to determine the importance of each feature
    shap_importance = np.abs(shap_values.values).mean(axis=0)

    # Create a DataFrame to view the feature importances
    feature_importance_df = pd.DataFrame({
        'Feature': x_train.columns,
        'SHAP Importance': shap_importance
    })

    # Return the DataFrame sorted by importance
    return feature_importance_df.sort_values(by='SHAP Importance', ascending=False)

