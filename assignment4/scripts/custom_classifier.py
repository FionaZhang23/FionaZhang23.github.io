from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted

import numpy as np

class OneHotClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_classifier=None):
        if base_classifier is None:
            base_classifier = RandomForestClassifier(n_estimators=100, max_depth=15)
        self.base_classifier = base_classifier
        self.encoder = OneHotEncoder(sparse_output=False, categories=[list(range(4))])

    def fit(self, X, y):
        num_images, H, W = X.shape[:3]
        num_pixels = num_images * H * W
        X = X.reshape(num_pixels, -1) 
        if y.shape != (num_pixels,):  # Reshape only if needed
            y = y.reshape(num_pixels)
        self.base_classifier.fit(X, y) 
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        num_images, H, W, C = X.shape
        num_pixels = num_images * H * W

        X = X.reshape(num_pixels, -1)

        labels = self.base_classifier.predict(X).reshape(-1, 1)
        one_hot_labels = self.encoder.fit_transform(labels)

        reshaped_output = one_hot_labels.reshape(num_images, H, W, 4) 
        return reshaped_output
