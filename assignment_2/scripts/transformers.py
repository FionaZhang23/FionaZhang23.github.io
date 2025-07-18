from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd  # Ensure Pandas is imported

from check_quality import detect_data_leakage

class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold = 0.9):
        self.threshold = threshold
        self.leaked_indices_ = [] 

    def fit(self, X, y=None):
        if not isinstance(X, np.ndarray):
            raise ValueError("Input X must be a NumPy array.")

        correlations = []
        for i in range(X.shape[1]):
            feature = X[:, i]
            if np.std(feature) == 0 or np.std(y) == 0:
                corr = 0
            else:
                corr = np.corrcoef(feature, y)[0, 1]
            correlations.append(np.nan_to_num(abs(corr)))
        correlations = np.array(correlations)
        self.leaked_indices_ = np.where(correlations > self.threshold)[0].tolist()
        return self

    def transform(self, X):
        print(X.shape)
        if not isinstance(X, np.ndarray):
            raise ValueError("Input X must be a NumPy array.")
        if len(self.leaked_indices_) == 0:
            return X
        X_transformed = np.delete(X, self.leaked_indices_, axis=1)
        print(X_transformed.shape)
        return X_transformed