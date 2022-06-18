from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


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

    # split the data into cv folds
    x_folds = np.array_split(X, cv)
    y_folds = np.array_split(y, cv)

    train_score_arr = []
    validation_score_arr = []

    # fit on each fold and Callback on each fold
    for i in range(cv):
        # organize train_data and test_data
        x_temp = np.delete(x_folds, i, axis=0)
        train_data_x = np.concatenate(x_temp)
        test_data_x = x_folds[i]

        y_temp = np.delete(y_folds, i, axis=0)
        train_data_y = np.concatenate(y_temp)
        test_data_y = y_folds[i]

        # fit
        estimator.fit(train_data_x, train_data_y)

        # update score arrays
        train_score_arr.append(scoring(train_data_y, estimator.predict(train_data_x)))
        validation_score_arr.append(scoring(test_data_y, estimator.predict(test_data_x)))

    # return average train score and validation_score
    return np.array(train_score_arr).mean(), np.array(validation_score_arr).mean()












