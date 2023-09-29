import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score)
from math import sqrt
import os
import joblib

regression_models = [
    {
        'name': 'Linear Regression',
        'model': LinearRegression(),
        'params': {'fit_intercept': [True, False],
                   'copy_X': [True, False]}
    },
    {
        'name': 'Random Forest Regressor',
        'model': RandomForestRegressor(),
        'params': {'n_estimators': [50, 100, 200],
                   'max_depth': [None, 10, 20, 30],
                   'min_samples_split': [2, 5, 10],
                   'min_samples_leaf': [1, 2, 4]}
    },
    {
        'name': 'Gradient Boosting Regressor',
        'model': GradientBoostingRegressor(),
        'params': {'n_estimators': [50, 100, 200],
                   'learning_rate': [0.01, 0.1, 0.2],
                   'max_depth': [3, 4, 5]}
    },
    {
        'name': 'Support Vector Regressor',
        'model': SVR(),
        'params': {'C': [0.1, 1, 10],
                   'kernel': ['linear', 'rbf'],
                   'epsilon': [0.1, 0.2, 0.5]}
    },
    {
        'name': 'Ridge Regression',
        'model': Ridge(),
        'params': {'alpha': [0.1, 1, 10]}
    },
    {
        'name': 'Lasso Regression',
        'model': Lasso(),
        'params': {'alpha': [0.1, 1, 10]}
    },
    {
        'name': 'MLP Regressor',
        'model': MLPRegressor(),
        'params': {'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                   'activation': ['relu', 'tanh'],
                   'alpha': [0.0001, 0.001, 0.01]}
    }
]


def save_models(trained_models):
    # Create the "models" folder if it doesn't exist
    if not os.path.exists("models"):
        os.makedirs("models")

    for model_name, model in trained_models.items():
        # Remove spaces and create the file name
        file_name = model_name.replace(" ", "") + ".joblib"

        # Create the full path to save the model
        model_path = os.path.join("models", file_name)

        # Save the model to the specified path
        joblib.dump(model, model_path)


# TODO: BETTER SPLIT DATA
def split_data(features, target):
    return train_test_split(features, target, test_size=0.3)


def run_all(X_train, X_test, y_train, y_test):
    results = []
    trained_models = {}

    # Loop through each regression model
    for model_info in regression_models:
        model = model_info['model']
        params = model_info['params']
        model_name = model_info['name']

        grid_search = GridSearchCV(model, params, cv=5, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        trained_model = grid_search.best_estimator_
        best_params = str(grid_search.best_params_)
        y_pred = grid_search.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        results.append({
            'Model': model_name,
            'Best Parameters': best_params,
            'Best Score': grid_search.best_score_,
            'Mean Absolute Error': mae,
            'Mean Squared Error': mse,
            'Root Mean Squared Error': rmse,
            'R-squared': r2
        })
        trained_models[model_name] = trained_model
        print("Finished analysing " + model_name)
    results_df = pd.DataFrame(results)
    results_df.to_csv('results.csv', index=False)
    save_models(trained_models)
    return trained_models
