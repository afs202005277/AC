import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator


class CustomCrossValidator(BaseCrossValidator):
    def get_n_splits(self, X=None, y=None, groups=None):
        t1, t2 = self.split_data(X, y, 3, 7, 'EFF')
        return len(t1)

    def split_data(self, X_train, y_train, min_years, max_years, target_column):
        dataset = pd.concat([X_train, y_train], axis=1)
        years = dataset['year'].unique()
        X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []

        # Iterate through the dataset using a sliding window of years
        for window_size in range(min_years, max_years + 1):
            for i in range(
                    max(years) - window_size + 1):
                # Define the start and end years for the window
                start_year = i
                end_year = start_year + window_size

                # Extract the data for the sliding window
                window_data = dataset[(dataset['year'] >= start_year) & (dataset['year'] < end_year)]

                # Split the window data into features (X) and target (y)
                X_window = window_data.drop(columns=[target_column])
                y_window = window_data[target_column]

                # Append the current window data to the lists
                X_train_list.append(X_window)
                y_train_list.append(y_window)

                # Extract the data for the next year (outside the window)
                next_year_data = dataset[dataset['year'] == end_year]
                X_next_year = next_year_data.drop(columns=[target_column])
                y_next_year = next_year_data[target_column]

                X_test_list.append(X_next_year)
                y_test_list.append(y_next_year)
        return X_train_list, y_train_list

    def split(self, X=None, y=None, groups=None):
        groups = self.split_data(X, y, 3, 7, 'EFF')
        for X_train, y_train in zip(groups[0], groups[1]):
            yield np.arange(len(X_train)), np.arange(len(y_train))
