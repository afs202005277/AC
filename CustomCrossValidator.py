import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator


class CustomCrossValidator(BaseCrossValidator):

    def __init__(self, min_years, max_years, target_column):
        self.min_years = min_years
        self.max_years = max_years
        self.target_column = target_column

    def get_n_splits(self, x=None, y=None, groups=None):
        t1, t2 = self.split_data(x, y)
        return len(t1)

    def split_data(self, x_train, y_train):
        dataset = pd.concat([x_train, y_train], axis=1)
        years = dataset['year'].unique()
        x_train_list, x_test_list, y_train_list, y_test_list = [], [], [], []

        # Iterate through the dataset using a sliding window of years
        for window_size in range(self.min_years, self.max_years + 1):
            for i in range(
                    max(years) - window_size + 1):
                # Define the start and end years for the window
                start_year = i
                end_year = start_year + window_size

                # Extract the data for the sliding window
                window_data = dataset[(dataset['year'] >= start_year) & (dataset['year'] < end_year)]

                # Split the window data into features (X) and target (y)
                x_window = window_data.drop(columns=[self.target_column])
                y_window = window_data[self.target_column]

                # Append the current window data to the lists
                x_train_list.append(x_window)
                y_train_list.append(y_window)

                # Extract the data for the next year (outside the window)
                next_year_data = dataset[dataset['year'] == end_year]
                x_next_year = next_year_data.drop(columns=[self.target_column])
                y_next_year = next_year_data[self.target_column]

                x_test_list.append(x_next_year)
                y_test_list.append(y_next_year)
        return x_train_list, y_train_list

    def split(self, x=None, y=None, groups=None):
        groups = self.split_data(x, y)
        for X_train, y_train in zip(groups[0], groups[1]):
            yield np.arange(len(X_train)), np.arange(len(y_train))
