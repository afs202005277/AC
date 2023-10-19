import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score)
from math import sqrt
import os
import joblib
from CustomCrossValidator import CustomCrossValidator

regression_models = [
    {
        'name': 'Random Forest Regressor',
        'model': RandomForestRegressor(),
        'params': {'n_estimators': [50],
                   'max_depth': [None, 10],
                   'min_samples_split': [2, 5, 10],
                   'min_samples_leaf': [1, 2, 4]}
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


def run_all(x_train, y_train, x_test, y_test, min_years, max_years, target_column):
    results = []
    trained_models = {}

    # Loop through each regression model
    for model_info in regression_models:
        model = model_info['model']
        params = model_info['params']
        model_name = model_info['name']

        grid_search = GridSearchCV(model, params, cv=CustomCrossValidator(min_years, max_years, target_column), n_jobs=-1)
        grid_search.fit(x_train, y_train)
        trained_model = grid_search.best_estimator_
        best_params = str(grid_search.best_params_)
        y_pred = grid_search.predict(x_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        results.append({
            'Model': model_name + "_" + target_column,
            'Best Parameters': best_params,
            'Best Score': grid_search.best_score_,
            'Mean Absolute Error': mae,
            'Mean Squared Error': mse,
            'Root Mean Squared Error': rmse,
            'R-squared': r2
        })
        trained_models[model_name] = trained_model
        print("Finished analyzing " + model_name)

    results_df = pd.DataFrame(results)
    results_df.to_csv('results_' + target_column + '.csv', index=False)
    save_models(trained_models)
    return trained_models
