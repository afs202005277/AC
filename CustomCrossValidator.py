import numpy as np
from sklearn.model_selection import BaseCrossValidator


class CustomCrossValidator(BaseCrossValidator):
    def __init__(self, X_train_list, y_train_list):
        self.X_train_list = X_train_list
        self.y_train_list = y_train_list

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self.X_train_list)

    def split(self, X=None, y=None, groups=None):
        for X_train, y_train in zip(self.X_train_list, self.y_train_list):
            n_samples = len(X_train)
            indices = np.arange(n_samples)
            fold_size = n_samples // len(self.X_train_list)

            for i in range(len(self.X_train_list)):
                start = i * fold_size
                end = (i + 1) * fold_size
                test_indices = indices[start:end]
                train_indices = np.concatenate((indices[:start], indices[end:]))

                yield train_indices, test_indices
