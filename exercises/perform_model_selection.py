from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def best_valid(est, list, train_X, train_y, test_X ,test_y, range):
    min_ind = np.argmin(np.array(list))
    selected_k = np.array(range)[min_ind]
    selected_error = list[min_ind]
    poli = est(selected_k)
    poli.fit(train_X, train_y)
    print("Best parameter -", selected_k)
    print("MSE -", round(mean_square_error(poli.predict(test_X), test_y),0))


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
    response = lambda x: (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    X = np.linspace(-1.2, 2, n_samples)
    y = response(X) + np.random.normal(0, noise, n_samples)
    train_X, train_y, test_X, test_y = split_train_test(pd.DataFrame(X), pd.Series(y), 2/3)

    make_subplots(rows=1, cols=1) \
        .add_traces(
        [go.Scatter(x=X, y=response(X), mode='lines', marker=dict(color="black"), showlegend=False),
         go.Scatter(x=test_X[0], y=test_y, mode='markers', marker=dict(color="blue"), name="Test"),
         go.Scatter(x=train_X[0], y=train_y, mode='markers', marker=dict(color="red"),
                    name="Train")],
        rows=1, cols=1) \
        .update_layout(title_text="Train and Test Samples", height=350, width=650) \
        .show()


    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,
    k_range = list(range(11))
    list_validation = []
    list_training = []
    for i in range(11):
        poli =PolynomialFitting(i)
        tra, vali = cross_validate(poli, train_X[0].to_numpy(), train_y.to_numpy(), mean_square_error, 5)
        list_training.append(tra)
        list_validation.append(vali)

    go.Figure([go.Scatter(name='Train Error', x=k_range, y=list_training, mode='markers+lines',
                          marker_color='rgb(152,171,150)'),
               go.Scatter(name='Validation Error', x=k_range, y=list_validation, mode='markers+lines',
                          marker_color='rgb(25,115,132)'),
               ]) \
        .update_layout(title="The average training and validation errors of poly of degree k",
                       xaxis_title="Degree k",
                       yaxis_title="MSE", height=350, width=650).show()





    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    min_ind = np.argmin(np.array(list_validation))
    selected_k = np.array(k_range)[min_ind]
    selected_error = list_validation[min_ind]
    poli = PolynomialFitting(selected_k)
    poli.fit(train_X[0].to_numpy(), train_y.to_numpy())
    print("Best parameter -", selected_k)
    print("MSE -", round(mean_square_error(poli.predict(test_X[0].to_numpy()), test_y.to_numpy()),2))








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
    X_train, y_train = X[:n_samples, ], y[:n_samples]
    X_test, y_test = X[n_samples:, ], y[n_samples:]

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    domaina = np.linspace(0.001, 10, n_evaluations)
    err_ridge_train = []
    err_ridge_val = []
    for i in domaina:##ridge
        ridge_est = RidgeRegression(i)
        a, b = cross_validate(ridge_est, X_train, y_train, mean_square_error, 5)
        err_ridge_train.append(a)
        err_ridge_val.append(b)

    err_lasso_train = []
    err_lasso_val = []
    domainb = np.linspace(0.001, 5, n_evaluations)
    for i in domainb:##lasso
        lasso_est = Lasso(i)
        c, d = cross_validate(lasso_est, X_train, y_train, mean_square_error, 5)
        err_lasso_train.append(c)
        err_lasso_val.append(d)

    go.Figure([go.Scatter(name='Train Error', x=domaina, y=err_ridge_train, mode='markers+lines',
                          marker_color='rgb(152,171,150)'),
               go.Scatter(name='Validation Error', x=domaina, y=err_ridge_val, mode='markers+lines',
                          marker_color='rgb(25,115,132)'),
               ]) \
        .update_layout(title=r"$\text{ }\text{Average training and Validation errors}$",
                       xaxis_title=r"$k\text{ -  Polynomial Degree}$",
                       yaxis_title=r"$\text{Error Value}$", height=650, width=850).show()

    go.Figure([go.Scatter(name='Train Error', x=domainb, y=err_lasso_train, mode='markers+lines',
                          marker_color='rgb(152,171,150)'),
               go.Scatter(name='Validation Error', x=domainb, y=err_lasso_val, mode='markers+lines',
                          marker_color='rgb(25,115,132)'),
               ]) \
        .update_layout(title=r"$\text{ }\text{Average training and Validation errors}$",
                       xaxis_title=r"$k\text{ -  Polynomial Degree}$",
                       yaxis_title=r"$\text{Error Value}$", height=650, width=850).show()



    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    print("Ridge-")
    best_valid(RidgeRegression, err_ridge_val, X_train, y_train, X_test, y_test, domaina)##ridge check
    print("*******")
    print("Lasso-")
    best_valid(Lasso, err_lasso_val, X_train, y_train, X_test, y_test, domainb)##lasso check
    ##linear regression check
    print("*******")
    print("Linear regression-")
    poli = LinearRegression()
    poli.fit(X_train, y_train)
    print("MSE -", round(mean_square_error(poli.predict(X_test), y_test),2))


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree(100, 5)
    select_polynomial_degree(100, 0)
    select_polynomial_degree(1500, 10)
    select_regularization_parameter()
