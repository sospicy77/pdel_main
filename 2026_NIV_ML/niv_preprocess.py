"""Preprocessing helpers (kept in a separate module so saved models unpickle cleanly)."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class LogColumns(BaseEstimator, TransformerMixin):
    """Apply log1p to selected columns (e.g. right-skewed precipitation)."""

    def __init__(self, cols=(3, 6)):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.array(X, dtype=float, copy=True)
        X[:, list(self.cols)] = np.log1p(X[:, list(self.cols)])
        return X
