import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.metrics import misclassification_error
from IMLearn.model_selection import cross_validate
from IMLearn.utils import split_train_test
from sklearn.metrics import roc_curve, auc
from utils import custom



import plotly.graph_objects as go


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines",
                                 marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}", width=600, height=400))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """

    def callback(weights, val, solver=None, grad=None, t=None, eta=None, delta=None):
        weights_list.append(weights)
        values_list.append(val)
        return

    values_list = []
    weights_list = []
    return callback, values_list, weights_list


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    l1 = L1(init)
    l2 = L2(init)
    callback, vals, weights = get_gd_state_recorder_callback()
    callback1, vals1, weights1 = get_gd_state_recorder_callback()
    grad_a = GradientDescent(callback=callback, learning_rate=FixedLR(0.01))
    grad_b = GradientDescent(callback=callback1, learning_rate=FixedLR(0.01))
    grad_a.fit(f=l1, X=None, y=None)
    grad_b.fit(f=l2, X=None, y=None)
    new = np.stack(weights, axis=0)
    new1 = np.stack(weights1, axis=0)
    fig = plot_descent_path(module=L1, descent_path=new, title="L1 Module")
    fig.show()
    fig = plot_descent_path(module=L2, descent_path=new1, title="L2 Module")
    fig.show()

    # Q2
    # L1
    k_range=list(range(len(vals)))
    l1_1 = L1(init)
    l1_01 = L1(init)
    l1_001 = L1(init)
    l1_0001 = L1(init)
    callback1, vals1, weights1 = get_gd_state_recorder_callback()
    callback01, vals01, weights01 = get_gd_state_recorder_callback()
    callback001, vals001, weights001 = get_gd_state_recorder_callback()
    callback0001, vals0001, weights0001 = get_gd_state_recorder_callback()
    grad_1 = GradientDescent(callback=callback1, learning_rate=FixedLR(1))
    grad_01 = GradientDescent(callback=callback01, learning_rate=FixedLR(0.1))
    grad_001 = GradientDescent(callback=callback001, learning_rate=FixedLR(0.01))
    grad_0001 = GradientDescent(callback=callback0001, learning_rate=FixedLR(0.001))
    grad_1.fit(f=l1_1, X=None, y=None)
    grad_01.fit(f=l1_01, X=None, y=None)
    grad_001.fit(f=l1_001, X=None, y=None)
    grad_0001.fit(f=l1_0001, X=None, y=None)
    go.Figure([go.Scatter(name='eta=1', x=k_range, y=vals1, mode='markers+lines'),
               go.Scatter(name='eta=0.1', x=k_range, y=vals01, mode='markers+lines'),
               go.Scatter(name='eta=0.01', x=k_range, y=vals001, mode='markers+lines'),
               go.Scatter(name='eta=0.001', x=k_range, y=vals0001, mode='markers+lines')
               ]) \
        .update_layout(title="L1 Loss As function of itrerations",
                       xaxis_title="Iterations",
                       yaxis_title="Loss", height=500, width=900).show()

    # L2
    k_range = list(range(len(vals)))
    l2_1 = L2(init)
    l2_01 = L2(init)
    l2_001 = L2(init)
    l2_0001 = L2(init)
    callback12, vals12, weights12 = get_gd_state_recorder_callback()
    callback012, vals012, weights012 = get_gd_state_recorder_callback()
    callback0012, vals0012, weights0012 = get_gd_state_recorder_callback()
    callback00012, vals00012, weights00012 = get_gd_state_recorder_callback()
    grad_12 = GradientDescent(callback=callback12, learning_rate=FixedLR(1))
    grad_012 = GradientDescent(callback=callback012, learning_rate=FixedLR(0.1))
    grad_0012 = GradientDescent(callback=callback0012, learning_rate=FixedLR(0.01))
    grad_00012 = GradientDescent(callback=callback00012, learning_rate=FixedLR(0.001))
    grad_12.fit(f=l2_1, X=None, y=None)
    grad_012.fit(f=l2_01, X=None, y=None)
    grad_0012.fit(f=l2_001, X=None, y=None)
    grad_00012.fit(f=l2_0001, X=None, y=None)
    go.Figure([go.Scatter(name='eta=1', x=k_range, y=vals12, mode='markers+lines') ,
               go.Scatter(name='eta=0.1', x=k_range, y=vals012, mode='markers+lines'),
               go.Scatter(name='eta=0.01', x=k_range, y=vals0012, mode='markers+lines'),
               go.Scatter(name='eta=0.001', x=k_range, y=vals00012, mode='markers+lines')
               ]) \
        .update_layout(title="L2 Loss As function of itrerations",
                       xaxis_title="Iterations",
                       yaxis_title="Loss", height=500, width=900).show()



def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    l1_1 = L1(init)
    l1_01 = L1(init)
    l1_001 = L1(init)
    l1_0001 = L1(init)
    callback1, vals1, weights1 = get_gd_state_recorder_callback()
    callback01, vals01, weights01 = get_gd_state_recorder_callback()
    callback001, vals001, weights001 = get_gd_state_recorder_callback()
    callback0001, vals0001, weights0001 = get_gd_state_recorder_callback()
    grad_1 = GradientDescent(callback=callback1, learning_rate=ExponentialLR(0.1, 0.9))
    grad_01 = GradientDescent(callback=callback01, learning_rate=ExponentialLR(0.1, 0.95))
    grad_001 = GradientDescent(callback=callback001, learning_rate=ExponentialLR(0.1, 0.99))
    grad_0001 = GradientDescent(callback=callback0001, learning_rate=ExponentialLR(0.1, 1))
    grad_1.fit(f=l1_1, X=None, y=None)
    grad_01.fit(f=l1_01, X=None, y=None)
    grad_001.fit(f=l1_001, X=None, y=None)
    grad_0001.fit(f=l1_0001, X=None, y=None)

    # Plot algorithm's convergence for the different values of gamma


    k_range = list(range(1000))
    go.Figure([go.Scatter(name='eta=1', x=k_range, y=vals1, mode='markers+lines', marker_color='rgb(152,171,150)'),
               go.Scatter(name='eta=0.1', x=k_range, y=vals01, mode='markers+lines', marker_color='rgb(220,179,144)'),
               go.Scatter(name='eta=0.01', x=k_range, y=vals001, mode='markers+lines', marker_color='rgb(25,115,132)'),
               go.Scatter(name='eta=0.001', x=k_range, y=vals0001, mode='markers+lines', marker_color='rgb(100,30,132)')
               ]) \
        .update_layout(title="L1 Loss As function of itrerations",
                       xaxis_title="Iterations",
                       yaxis_title="Loss", height=500, width=900).show()
    # Plot descent path for gamma=0.95
    l1 = L1(init)
    callback, vals, weights = get_gd_state_recorder_callback()
    grad_a = GradientDescent(callback=callback, learning_rate=ExponentialLR(0.1, 0.95))
    grad_a.fit(f=l1, X=None, y=None)
    new = np.stack(weights, axis=0)
    fig = plot_descent_path(module=L1, descent_path=new, title="L1 Module")
    fig.show()


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # Plotting convergence rate of logistic regression over SA heart disease data
    log = LogisticRegression(include_intercept=True, solver=GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000), penalty="none", lam= 1, alpha=.5)
    log.fit(pd.DataFrame.to_numpy(X_train), pd.Series.to_numpy(y_train))
    fpr, tpr, thresholds = roc_curve(pd.Series.to_numpy(y_train), log.predict_proba(pd.DataFrame.to_numpy(X_train)))
    c = [custom[0], custom[-1]]
    go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds, name="", showlegend=False, marker_size=5,marker_color=c[1][1],

                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
                         xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                         yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$"), height=500, width=900)).show()
    best_alpha = thresholds[np.argmax(tpr - fpr)]
    print(f"Best alpha :{best_alpha}")
    log.alpha_ = best_alpha
    test_error = log._loss(pd.DataFrame.to_numpy(X_test), pd.Series.to_numpy(y_test))
    print(f"Model's test error {test_error}")


    l1_model = LogisticRegression(include_intercept=True, penalty="l1",
                                  solver=GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20), lam=0)
    list = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    validation_errors = []
    for lam in list:
        estimator = LogisticRegression(include_intercept=True, penalty="l1",
                                       solver=GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20), lam=lam)
        train_err, v_error = cross_validate(estimator, pd.DataFrame.to_numpy(X_train), pd.Series.to_numpy(y_train), misclassification_error)
        validation_errors.append(v_error)

    best_lambda = list[np.argmin(np.array(validation_errors))]
    l1_model.lam_ = best_lambda
    l1_model.fitted_ = True

    print(f"Best l1 lambda  {best_lambda}")
    test_error = l1_model.loss(pd.DataFrame.to_numpy(X_test), pd.Series.to_numpy(y_test))
    print(f"L1 model's test error  {test_error}")

    # L2

    l2_model = LogisticRegression(include_intercept=True, penalty="l2",
                                  solver=GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20), lam=0)
    l2_model.fitted_ = True
    validation_errors = []
    for lam in list:
        estimator = LogisticRegression(include_intercept=True, penalty="l1",
                                       solver=GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20), lam=lam)
        train_err, v_error = cross_validate(estimator,  pd.DataFrame.to_numpy(X_train), pd.Series.to_numpy(y_train), misclassification_error)
        validation_errors.append(v_error)
    best_lambda = list[np.argmin(np.array(validation_errors))]
    l2_model.lam_ = best_lambda
    print(f"Best l2 lambda {best_lambda}")
    test_error = l2_model._loss( pd.DataFrame.to_numpy(X_test), pd.Series.to_numpy(y_test))
    print(f"L2 model's test error{test_error}")



if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    # fit_logistic_regression()
