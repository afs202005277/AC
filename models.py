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

fast_regression_models = [
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
    }
]

fast_classifier_models = [
    {
        'name': 'Linear SVC',
        'model': LinearSVC(),
        'params': {'C': [0.1, 1, 10],
                   'penalty': ['l1', 'l2']}
    }
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
        results_df.to_csv('results' + name + '_' + target_column + '.csv', index=False)
        save_models(trained_models, name, target_column)

    return dict([(x[0], y) for x, y in trained_models.items()])


"""
75 fits failed with the following error:
Traceback (most recent call last):
  File "/usr/local/lib/python3.10/dist-packages/sklearn/model_selection/_validation.py", line 729, in _fit_and_score
    estimator.fit(X_train, y_train, **fit_params)
  File "/usr/local/lib/python3.10/dist-packages/sklearn/base.py", line 1152, in wrapper
    return fit_method(estimator, *args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/sklearn/svm/_classes.py", line 326, in fit
    self.coef_, self.intercept_, n_iter_ = _fit_liblinear(
  File "/usr/local/lib/python3.10/dist-packages/sklearn/svm/_base.py", line 1229, in _fit_liblinear
    solver_type = _get_liblinear_solver_type(multi_class, penalty, loss, dual)
  File "/usr/local/lib/python3.10/dist-packages/sklearn/svm/_base.py", line 1060, in _get_liblinear_solver_type
    raise ValueError(
ValueError: Unsupported set of arguments: The combination of penalty='l1' and loss='squared_hinge' are not supported when dual=True, Parameters: penalty='l1', loss='squared_hinge', dual=True

  warnings.warn(some_fits_failed_message, FitFailedWarning)
/usr/local/lib/python3.10/dist-packages/sklearn/model_selection/_search.py:979: UserWarning: One or more of the test scores are non-finite: [       nan 0.7920258         nan 0.74323316        nan 0.65616639]
  warnings.warn(
/usr/local/lib/python3.10/dist-packages/sklearn/svm/_classes.py:32: FutureWarning: The default value of `dual` will change from `True` to `'auto'` in 1.5. Set the value of `dual` explicitly to suppress the warning.
  warnings.warn(
/usr/local/lib/python3.10/dist-packages/sklearn/svm/_base.py:1250: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Traceback (most recent call last):
  File "/home/andre/Desktop/AC/models.py", line 181, in run_all
    trained_models[(model_name, scaler_name)] = joblib.load(
  File "/usr/local/lib/python3.10/dist-packages/joblib/numpy_pickle.py", line 650, in load
    with open(filename, 'rb') as f:
FileNotFoundError: [Errno 2] No such file or directory: 'models/EnsembleClassifier_Teams_None_playoff.joblib'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/andre/Desktop/AC/main.py", line 804, in <module>
    main()
  File "/home/andre/Desktop/AC/main.py", line 741, in main
    trained_models_teams = models_train_and_test_teams(dataframes_dict['teams'], features, target)
  File "/home/andre/Desktop/AC/main.py", line 441, in models_train_and_test_teams
    trained_models = models.run_all(x_train, y_train, x_test, y_test, 3, 7, target, "Teams", FAST)
  File "/home/andre/Desktop/AC/models.py", line 192, in run_all
    grid_search.fit(x_train, y_train)
  File "/usr/local/lib/python3.10/dist-packages/sklearn/base.py", line 1152, in wrapper
    return fit_method(estimator, *args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/sklearn/model_selection/_search.py", line 898, in fit
    self._run_search(evaluate_candidates)
  File "/usr/local/lib/python3.10/dist-packages/sklearn/model_selection/_search.py", line 1422, in _run_search
    evaluate_candidates(ParameterGrid(self.param_grid))
  File "/usr/local/lib/python3.10/dist-packages/sklearn/model_selection/_search.py", line 875, in evaluate_candidates
    _warn_or_raise_about_fit_failures(out, self.error_score)
  File "/usr/local/lib/python3.10/dist-packages/sklearn/model_selection/_validation.py", line 414, in _warn_or_raise_about_fit_failures
    raise ValueError(all_fits_failed_message)
ValueError: 
All the 50 fits failed.
It is very likely that your model is misconfigured.
You can try to debug the error by setting error_score='raise'.

Below are more details about the failures:
--------------------------------------------------------------------------------
50 fits failed with the following error:
Traceback (most recent call last):
  File "/usr/local/lib/python3.10/dist-packages/sklearn/model_selection/_validation.py", line 729, in _fit_and_score
    estimator.fit(X_train, y_train, **fit_params)
  File "/usr/local/lib/python3.10/dist-packages/sklearn/base.py", line 1152, in wrapper
    return fit_method(estimator, *args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_voting.py", line 349, in fit
    return super().fit(X, transformed_y, sample_weight)
  File "/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_voting.py", line 73, in fit
    names, clfs = self._validate_estimators()
  File "/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_base.py", line 282, in _validate_estimators
    raise ValueError(
ValueError: The estimator Lasso should be a classifier.
"""
