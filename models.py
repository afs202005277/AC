import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, Normalizer
from sklearn.svm import SVR, LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
from sklearn.naive_bayes import GaussianNB

from math import sqrt
import os
import joblib
from sklearn.tree import DecisionTreeClassifier

from CustomCrossValidator import CustomCrossValidator

slow_regression_models = [
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
        'params': {'C': [0.1, 1, 10, 100, 1000],
                   'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                   'degree': [2, 3, 4, 5],
                   'gamma': ['scale', 'auto'],
                   'coef0': [0, 1, 2, 3],
                   'shrinking': [True, False],
                   'tol': [1e-3, 1e-4, 1e-5],
                   'max_iter': [1000, 2000, 25000]}
    },
    {
        'name': 'Ridge Regression',
        'model': Ridge(),
        'params': {'alpha': [0.1, 1, 10]}
    },
    {
        'name': 'Lasso Regression',
        'model': Lasso(),
        'params': {
            'alpha': [0.1, 1, 10],
            'tol': [1e-4, 1e-5, 1e-6],
            'max_iter': [5000, 7500, 10000]
        }
    },
    {
        'name': 'MLP Regressor',
        'model': MLPRegressor(),
        'params': {
            'hidden_layer_sizes': [(100,), (50, 100), (50, 50, 100), (100, 100)],
            'activation': ['relu', 'tanh', 'logistic'],
            'solver': ['sgd', 'adam'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive'],
            'max_iter': [200, 400, 600],
            'early_stopping': [True, False]
        }
    }
]
"""
A list of regression models along with their associated hyperparameter search spaces.
Each model is represented as a dictionary containing the model's name, the model instance, and a dictionary of hyperparameter values to be explored.
"""

slow_classifier_models = [
    {
        'name': 'Linear SVC',
        'model': LinearSVC(),
        'params': {'C': [0.1, 1, 10],
                   'penalty': ['l1', 'l2']}
    },
    {
        'name': 'Naive Bayes',
        'model': GaussianNB(),
        'params': {}
    },
    {
        'name': 'KNeighbors Classifier',
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': [1, 3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'leaf_size': [10, 20, 30, 40, 50],
            'p': [1, 2],
            'metric': ['euclidean', 'manhattan', 'chebyshev']
        }
    },
    {
        'name': 'SVC',
        'model': SVC(),
        'params': {'C': [0.1, 1, 10, 100, 1000],
                   'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                   'degree': [2, 3, 4, 5],
                   'gamma': ['scale', 'auto'],
                   'coef0': [0, 1, 2, 3],
                   'shrinking': [True, False],
                   'tol': [1e-3, 1e-4, 1e-5],
                   'max_iter': [1000, 2000, 25000],
                   'decision_function_shape': ['ovr', 'ovo']}
    },
    {
        'name': 'Ensemble Classifier',
        'model': VotingClassifier(estimators=[('lr', LogisticRegression()), ('rf', RandomForestClassifier()), ('dt',
                                                                                                               DecisionTreeClassifier(
                                                                                                                   criterion='gini',
                                                                                                                   splitter='best',
                                                                                                                   max_depth=None,
                                                                                                                   min_samples_split=5,
                                                                                                                   min_samples_leaf=5,
                                                                                                                   max_features=None,
                                                                                                                   random_state=42))]),
        'params': {'voting': ['hard', 'soft']}
    }
]
"""
A list of classifier models along with their associated hyperparameter search spaces.
Each model is represented as a dictionary containing the model's name, the model instance, and a dictionary of hyperparameter values to be explored.
"""

scalers = {'None': None, 'StandardScaler': StandardScaler(), 'MinMaxScaler': MinMaxScaler(),
           'RobustScaler': RobustScaler(), 'MaxAbsScaler': MaxAbsScaler(), 'Normalizer': Normalizer()}
"""
A dictionary containing different scalers and their corresponding instances.
Scalers are used for preprocessing input data before feeding it into machine learning models.
"""


def build_file_name(model_name, name, target_col, scaler_name):
    """
        Build a standardized file name for saving a machine learning model.

        Parameters:
        model_name (str): The name of the machine learning model.
        name (str): The identifier for the specific use or purpose of the model.
        target_col (str): The target column or variable that the model predicts.
        scaler_name (str): The name of the scaler used for preprocessing input data.

        Returns:
        str: The constructed file name for saving the model.
        """
    file_name = model_name.replace(" ", "") + '_' + name + '_' + target_col + '_' + scaler_name + ".joblib"
    return os.path.join("models", file_name)


def save_models(trained_models, name, target_col):
    """
        Save trained machine learning models to files.

        Parameters:
        trained_models (dict): A dictionary containing trained models along with their names and scalers.
                              Keys are tuples (model_name, scaler_name), and values are the corresponding models.
        name (str): The identifier for the specific use or purpose of the models.
        target_col (str): The target column or variable that the models predict.
        """
    if not os.path.exists("models"):
        os.makedirs("models")

    for (model_name, scaler_name), model in trained_models.items():
        # Remove spaces and create the file name
        model_path = build_file_name(model_name, name, scaler_name, target_col)
        # Save the model to the specified path
        joblib.dump(model, model_path)


def scale_dataframe(scaler, x_train, x_test):
    """
        Scale the features of a DataFrame using a specified scaler.

        Parameters:
        scaler: An instance of a scaler from scikit-learn or None if no scaling is needed.
        x_train (pd.DataFrame): The training features DataFrame.
        x_test (pd.DataFrame): The testing features DataFrame.

        Returns:
        pd.DataFrame, pd.DataFrame: Scaled training and testing features DataFrames.
        """
    columns = x_train.columns
    if scaler is not None:
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        x_train = pd.DataFrame(data=x_train, columns=columns)
        x_test = pd.DataFrame(data=x_test, columns=columns)

    return x_train, x_test


def run_all(x_train_original, y_train_original, x_test_original, y_test_original, min_years, max_years, target_column,
            name):
    """
       Run a comprehensive machine learning analysis including training, testing, and evaluation of models.

       Args:
       x_train_original (pd.DataFrame): The original training features DataFrame.
       y_train_original (pd.Series): The original training target variable.
       x_test_original (pd.DataFrame): The original testing features DataFrame.
       y_test_original (pd.Series): The original testing target variable.
       min_years (int): The minimum year for cross-validation.
       max_years (int): The maximum year for cross-validation.
       target_column (str): The target column or variable that the models predict.
       name (str): The identifier for the specific use or purpose of the models.

       Returns:
       dict: A dictionary containing trained machine learning models.
       """
    global scalers
    results = []
    trained_models = {}

    store = False
    regression_models = slow_regression_models
    classifier_models = slow_classifier_models

    if target_column == 'playoff':
        # Using Classifier Models on playoff
        for scaler_name, scaler in scalers.items():
            for model_info in classifier_models:
                x_train = x_train_original.copy()
                y_train = y_train_original.copy()
                x_test = x_test_original.copy()
                y_test = y_test_original.copy()
                model = model_info['model']
                params = model_info['params']
                model_name = model_info['name']

                y_train[y_train == 1] = 'Y'
                y_train[y_train == 0] = 'N'

                try:
                    trained_models[(model_name, scaler_name)] = joblib.load(
                        build_file_name(model_name, name, scaler_name, target_column))
                except FileNotFoundError:
                    store = True
                    cross_validator_elements = CustomCrossValidator(min_years, max_years, target_column).split(
                        x_train.copy(),
                        y_train.copy())
                    grid_search = GridSearchCV(model, params, cv=cross_validator_elements, n_jobs=-1)
                    x_train.drop('year', axis=1, inplace=True)
                    x_test.drop('year', axis=1, inplace=True)
                    x_train, x_test = scale_dataframe(scaler, x_train, x_test)
                    grid_search.fit(x_train, y_train)
                    trained_model = grid_search.best_estimator_
                    best_params = str(grid_search.best_params_)
                    y_pred = grid_search.predict(x_test)

                    y_test[y_test == 1] = 'Y'
                    y_test[y_test == 0] = 'N'

                    data = {'y_test': y_test, 'y_pred': y_pred}
                    data = pd.DataFrame(data)
                    data = data.sort_values(by='y_pred', ascending=False)
                    data = data.reset_index(drop=True)
                    data['y_pred'] = data['y_pred'].astype(str)
                    data.loc[:8, 'y_pred'] = 'Y'
                    data.loc[8:, 'y_pred'] = 'N'

                    y_test = data['y_test']
                    y_pred = data['y_pred']

                    print(str(data))

                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, pos_label='Y')
                    recall = recall_score(y_test, y_pred, pos_label='Y')
                    f1 = f1_score(y_test, y_pred, pos_label='Y')
                    conf_matrix = confusion_matrix(y_test, y_pred)
                    print(conf_matrix)
                    results.append({
                        'Model': model_name,
                        'Scaler': scaler_name,
                        'Best Parameters': best_params,
                        'Best Score': grid_search.best_score_,
                        'Mean Absolute Error': None,
                        'Mean Squared Error': None,
                        'Root Mean Squared Error': None,
                        'R-squared': None,
                        'Accuracy': accuracy,
                        'Recall': recall,
                        'Precision': precision,
                        'F1 Score': f1
                    })

                    trained_models[(model_name, scaler_name)] = trained_model
                    print("Finished analyzing " + model_name + " with " + scaler_name)

    for scaler_name, scaler in scalers.items():
        for model_info in regression_models:
            x_train = x_train_original.copy()
            y_train = y_train_original.copy()
            x_test = x_test_original.copy()
            y_test = y_test_original.copy()
            model = model_info['model']
            params = model_info['params']
            model_name = model_info['name']

            try:
                trained_models[(model_name, scaler_name)] = joblib.load(
                    build_file_name(model_name, name, scaler_name, target_column))
            except FileNotFoundError:
                store = True
                cross_validator_elements = CustomCrossValidator(min_years, max_years, target_column).split(
                    x_train.copy(),
                    y_train.copy())
                grid_search = GridSearchCV(model, params, cv=cross_validator_elements, n_jobs=-1)
                x_train.drop('year', axis=1, inplace=True)
                x_test.drop('year', axis=1, inplace=True)
                x_train, x_test = scale_dataframe(scaler, x_train, x_test)
                grid_search.fit(x_train, y_train)
                trained_model = grid_search.best_estimator_
                best_params = str(grid_search.best_params_)
                y_pred = grid_search.predict(x_test)

                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = sqrt(mse)
                r2 = r2_score(y_test, y_pred)

                if target_column == 'playoff':
                    y_test[y_test == 1] = 'Y'
                    y_test[y_test == 0] = 'N'

                    data = {'y_test': y_test, 'y_pred': y_pred}
                    data = pd.DataFrame(data)
                    data = data.sort_values(by='y_pred', ascending=False)
                    data = data.reset_index(drop=True)
                    data['y_pred'] = data['y_pred'].astype(str)
                    data.loc[:8, 'y_pred'] = 'Y'
                    data.loc[8:, 'y_pred'] = 'N'

                    y_test = data['y_test']
                    y_pred = data['y_pred']

                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, pos_label='Y')
                    recall = recall_score(y_test, y_pred, pos_label='Y')
                    f1 = f1_score(y_test, y_pred, pos_label='Y')
                    conf_matrix = confusion_matrix(y_test, y_pred)
                    print(conf_matrix)
                    results.append({
                        'Model': model_name,
                        'Scaler': scaler_name,
                        'Best Parameters': best_params,
                        'Best Score': grid_search.best_score_,
                        'Mean Absolute Error': mae,
                        'Mean Squared Error': mse,
                        'Root Mean Squared Error': rmse,
                        'R-squared': r2,
                        'Accuracy': accuracy,
                        'Recall': recall,
                        'Precision': precision,
                        'F1 Score': f1
                    })
                else:
                    results.append({
                        'Model': model_name,
                        'Scaler': scaler_name,
                        'Best Parameters': best_params,
                        'Best Score': grid_search.best_score_,
                        'Mean Absolute Error': mae,
                        'Mean Squared Error': mse,
                        'Root Mean Squared Error': rmse,
                        'R-squared': r2
                    })
                trained_models[(model_name, scaler_name)] = trained_model
                print("Finished analyzing " + model_name + " with " + scaler_name)

    if store:
        results_df = pd.DataFrame(results)
        results_df.to_csv('results' + name + '_' + target_column + '.csv', index=False)
        save_models(trained_models, name, target_column)

    return dict([(x[0] + "_" + x[1], y) for x, y in trained_models.items()])
