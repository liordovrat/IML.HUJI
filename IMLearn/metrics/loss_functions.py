import numpy as np


def mean_square_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate MSE loss

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    MSE of given predictions
    """
    return np.square(y_true-y_pred).mean()


def misclassification_error(y_true: np.ndarray, y_pred: np.ndarray, normalize: bool = True) -> float:
    """
    Calculate misclassification loss

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values
    normalize: bool, default = True
        Normalize by number of samples or not

    Returns
    -------
    Misclassification of given predictions
    """

    if y_true.size == 0:
        raise ValueError("y_true size 0")

    mis_counter = np.sum(y_true != y_pred)
    if normalize:
        return mis_counter / y_true.size
    return float(mis_counter)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy of given predictions

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    Accuracy of given predictions
    """
    if y_true.size == 0:
        raise ValueError("y_true size 0")

    return np.sum(y_true == y_pred) / y_true.size

def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the cross entropy of given predictions

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    Cross entropy of given predictions
    """
    raise NotImplementedError()
