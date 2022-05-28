from typing import NoReturn, List, Any
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier
    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`
    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`
    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`
    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`
    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for
        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_, classes_count = np.unique(y, return_counts= True)

        arr_mu = []
        self.cov_ = 0
        for i, k in enumerate(self.classes_):
            arr_mu.append(X[y == k].mean(axis=0))
            self.cov_ += (X[y == k] - np.asarray(arr_mu)[-1]).T @ (X[y == k] - np.asarray(arr_mu)[-1])

        self.mu_ = np.asarray(arr_mu)
        self.cov_ = self.cov_ / (X.shape[0] - self.classes_.size)

        self._cov_inv = inv(self.cov_)

        self.pi_ = classes_count / y.size


    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for
        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        matrix = self.likelihood(X)
        pred = []
        for Xi in matrix:
            pred.append(self.classes_[np.argmax(Xi)])

        return np.asarray(pred)


    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.
        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        likelihood = []
        a = np.sqrt(det(self.cov_) * (2 * np.pi) ** X.shape[1])
        for i in range(self.classes_.size):
            x_mu = X - self.mu_[i]
            exp = self.pi_[i] * np.exp(-0.5 * np.einsum('ij,ji->i', x_mu @ self._cov_inv, x_mu.T))
            likelihood.append(exp / a)
        return np.array(likelihood).T


    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples
        y : ndarray of shape (n_samples, )
            True labels of test samples
        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        return misclassification_error(y, self._predict(X))