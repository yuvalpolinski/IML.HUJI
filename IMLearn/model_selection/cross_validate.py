from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator
import pandas as pd


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    avg_train_err = 0.0
    avg_validation_err = 0.0
    groups = np.remainder(np.arange(y.size), cv)
    for k in range(cv):
        train_sets_x, train_sets_y = X[groups != k], y[groups != k]
        val_set_x, val_set_y = X[groups == k], y[groups == k]
        estimator.fit(train_sets_x, train_sets_y)
        loss_train_d = scoring(estimator.predict(train_sets_x), train_sets_y)
        loss_validation_d = scoring(estimator.predict(val_set_x), val_set_y)
        avg_train_err += loss_train_d / cv
        avg_validation_err += loss_validation_d / cv
    return (avg_train_err, avg_validation_err)
