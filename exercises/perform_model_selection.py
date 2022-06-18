from __future__ import annotations
import numpy as np
import pandas as pd
import sklearn.linear_model
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    X = np.linspace(-1.2, 2, n_samples)
    eps = np.random.normal(0, noise, n_samples)
    y = (X + 3) * (X + 2) * (X + 1) * (X - 1) * (X - 2)
    y_noise = y + eps

    train_X, train_y, test_X, test_y = split_train_test(pd.DataFrame(X), pd.Series(y_noise), 2 / 3)
    train_X, train_y, test_X, test_y = train_X.to_numpy()[:, 0], train_y.to_numpy(), test_X.to_numpy()[:,
                                                                                     0], test_y.to_numpy()

    fig = go.Figure(
        [go.Scatter(x=X, y=y, mode="lines+markers", marker=dict(color="black"), name=r"true noiseless"),
         go.Scatter(x=train_X, y=train_y, mode="markers", marker=dict(color="blue"), name=r"train set"),
         go.Scatter(x=test_X, y=test_y, mode="markers", marker=dict(color="red"), name=r"test set")]
    )
    fig.update_xaxes(title_text="x")
    fig.update_yaxes(title_text="y")
    fig.update_layout(
        title=f"Question 1 - true noiseless model, train set and test set, with number of samples: {n_samples} and "
              f"noise: {noise}")
    fig.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    train_score_arr, validation_score_arr = [], []

    # iterate the degrees
    for k in range(11):
        # for each degree calculate train and validation score
        train_score, validation_score = cross_validate(PolynomialFitting(k), train_X,
                                                       train_y, mean_square_error)
        train_score_arr.append(train_score)
        validation_score_arr.append(validation_score)

    fig = go.Figure(
        [go.Scatter(x=np.arange(0, 11), y=np.asarray(train_score_arr), mode='lines+markers', marker=dict(color="blue"),
                    name="train error"),
         go.Scatter(x=np.arange(0, 11), y=np.asarray(validation_score_arr), mode='lines+markers',
                    marker=dict(color="red"), name="validation error")]
    )
    fig.update_xaxes(title_text="polynom degree")
    fig.update_yaxes(title_text="error")
    fig.update_layout(
        title_text=f"Question 2 - Average training and validation errors for each of the polynomial degrees, "
                   f"with number of samples: {n_samples} and noise: {noise}")
    fig.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    k_min = np.argmin(validation_score_arr)
    p = PolynomialFitting(k_min)
    p.fit(train_X, train_y)
    pred = p.predict(test_X)
    test_err = mean_square_error(test_y, pred)

    print(f"For samples: {n_samples} and noise: {noise}, The best degree is: {k_min},"
          f" and the lowest validation error is: {validation_score_arr[k_min]}")
    print(f"The test error of the fitted model is: {np.round(test_err, 2)}")
    print("\n")


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    train_X, train_y, test_X, test_y = split_train_test(pd.DataFrame(X), pd.Series(y), n_samples / y.size)
    train_X, train_y, test_X, test_y = train_X.to_numpy(), train_y.to_numpy(), test_X.to_numpy(), test_y.to_numpy()

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    optional_lambdas = np.linspace(0.001, 3, n_evaluations)
    ridge_train_score_arr, ridge_validation_score_arr = [], []
    lasso_train_score_arr, lasso_validation_score_arr = [], []

    for lamb in optional_lambdas:
        # Cross_validate on Ridge
        ridge_train_score, ridge_validation_score = cross_validate(RidgeRegression(lamb), train_X, train_y,
                                                                   mean_square_error)
        ridge_train_score_arr.append(ridge_train_score)
        ridge_validation_score_arr.append(ridge_validation_score)

        # Cross_validate on Lasso
        lasso_train_score, lasso_validation_score = cross_validate(Lasso(alpha=lamb, max_iter=10000), train_X, train_y,
                                                                   mean_square_error)
        lasso_train_score_arr.append(lasso_train_score)
        lasso_validation_score_arr.append(lasso_validation_score)

    # plot
    fig = go.Figure(
        # Plot Ridge
        [go.Scatter(x=optional_lambdas, y=ridge_train_score_arr, mode='lines+markers',
                    marker=dict(color="blue"), name="Ridge train error"),
         go.Scatter(x=optional_lambdas, y=ridge_validation_score_arr, mode='lines+markers',
                    marker=dict(color="red"), name="Ridge validation error"),

         # Plot Lasso
         go.Scatter(x=optional_lambdas, y=lasso_train_score_arr, mode='lines+markers',
                    marker=dict(color="green"), name="Lasso train error"),
         go.Scatter(x=optional_lambdas, y=lasso_validation_score_arr, mode='lines+markers',
                    marker=dict(color="yellow"), name="Lasso validation error")]
    )

    fig.update_xaxes(title_text="lambda")
    fig.update_yaxes(title_text="error")
    fig.update_layout(
        title=f"Question 7 - Train and validation errors as a function of the regularization parameter")
    fig.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    # Ridge with the best lambda
    idx_min = np.argmin(ridge_validation_score_arr)
    ridge_lamb_min = optional_lambdas[idx_min]

    r = RidgeRegression(ridge_lamb_min)
    r.fit(train_X, train_y)
    test_err_ridge = r.loss(test_X, test_y)

    # Lasso with the best lambda
    idx_min = np.argmin(lasso_validation_score_arr)
    lasso_lamb_min = optional_lambdas[idx_min]

    lasso = Lasso(alpha=lasso_lamb_min, max_iter=50000)
    lasso.fit(train_X, train_y)
    test_err_lasso = mean_square_error(test_y, lasso.predict(test_X))

    lin_regression = LinearRegression()
    lin_regression.fit(train_X, train_y)
    test_err_lin_regression = lin_regression.loss(test_X, test_y)

    print(f"For Ridge: the best regularization value is: {ridge_lamb_min}, and the test error is: {test_err_ridge}")
    print(f"For Lasso: the best regularization value is: {lasso_lamb_min}, and the test error is: {test_err_lasso}")
    print("The test error of linear regression is: ", test_err_lin_regression)
    print("\n")

if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter()
