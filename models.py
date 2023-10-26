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
        'params': {'n_neighbors': [3, 5, 7],
                   'weights': ['uniform', 'distance']}
    },
    {
        'name': 'SVC',
        'model': SVC(),
        'params': {'C': [0.1, 1, 10],
                   'kernel': ['linear', 'rbf'],
                   'gamma': ['scale', 'auto']}
    },
    {
        'name': 'Ensemble Classifier',
        'model': VotingClassifier(estimators=[('lr', LogisticRegression()), ('rf', RandomForestClassifier()), ('gnb', Lasso(alpha=0.1))]),
        'params': {'voting': ['hard', 'soft']}
    }
]

fast_regression_models = [
    {
        'name': 'Random Forest Regressor',
        'model': RandomForestRegressor(),
        'params': {'n_estimators': [50, 100, 200],
                   'max_depth': [None, 10, 20, 30],
                   'min_samples_split': [2, 5, 10],
                   'min_samples_leaf': [1, 2, 4]}
    },
]

fast_classifier_models = [
    {
        'name': 'Linear SVC',
        'model': LinearSVC(),
        'params': {'C': [0.1, 1, 10],
                   'penalty': ['l1', 'l2']}
    },
]


scalers = {'None': None, 'StandardScaler': StandardScaler(), 'MinMaxScaler': MinMaxScaler(),
           'RobustScaler': RobustScaler(), 'MaxAbsScaler': MaxAbsScaler(), 'Normalizer': Normalizer()}


def build_file_name(model_name, name, target_col, scaler_name):
    file_name = model_name.replace(" ", "") + '_' + name + '_' + target_col + '_' + scaler_name + ".joblib"
    return os.path.join("models", file_name)


def save_models(trained_models, name, target_col):
    if not os.path.exists("models"):
        os.makedirs("models")

    for (model_name, scaler_name), model in trained_models.items():
        # Remove spaces and create the file name
        model_path = build_file_name(model_name, name, scaler_name, target_col)
        # Save the model to the specified path
        joblib.dump(model, model_path)


def scale_dataframe(scaler, x_train, x_test):
    columns = x_train.columns
    if scaler is not None:
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        x_train = pd.DataFrame(data=x_train, columns=columns)
        x_test = pd.DataFrame(data=x_test, columns=columns)

    return x_train, x_test


def run_all(x_train_original, y_train_original, x_test_original, y_test_original, min_years, max_years, target_column,
            name, fast):
    global scalers
    results = []
    trained_models = {}

    store = False
    if fast:
        regression_models = fast_regression_models
        classifier_models = fast_classifier_models
        scalers = {'None': scalers['None']}
    else:
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
                trained_models[(model_name, scaler_name)] = joblib.load(build_file_name(model_name, name, scaler_name, target_column))
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
        results_df.to_csv('results' + name + '.csv', index=False)
        save_models(trained_models, name, target_column)

    return dict([(x[0], y) for x, y in trained_models.items()])
