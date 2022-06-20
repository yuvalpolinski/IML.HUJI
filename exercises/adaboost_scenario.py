import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots



def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)
    # Question 1: Train- and test errors of AdaBoost in noiseless case
    ada = AdaBoost(DecisionStump, n_learners)
    ada.fit(train_X, train_y)
    test_err = []
    train_err = []
    for i in range(1, n_learners):
        test_err.append(ada.partial_loss(test_X, test_y, i))
        train_err.append(ada.partial_loss(train_X, train_y, i))

    x = list(range(n_learners))
    go.Figure([
        go.Scatter(x=x, y=test_err, mode='markers + lines', name=r'test_err'),
        go.Scatter(x=x, y=train_err, mode='markers + lines', name=r'train_err')]) \
        .update_layout(title=rf"$training- and test errors as a function of the number of fitted learners$", width=900, height=500).show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    fig = make_subplots(rows=2, cols=2, subplot_titles=[rf"$iterations={k}$" for k in T])
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    errors = []
    for i, T in enumerate([5, 50, 100, 250]):
        errors.append(ada.partial_loss(test_X,test_y, T))
        fig.add_traces(
            [decision_surface(lambda x: ada.partial_predict(x, T), lims[0], lims[1], showscale=False),
               go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                        marker=dict(color=test_y, colorscale=[custom[0], custom[-1]],
                                    line=dict(color="black", width=2)))], rows=(i // 2) + 1, cols=(i % 2) + 1)
    fig.update_layout(title="Decision Boundaries", width=600, height=600, margin=dict(t=100)).update_xaxes(visible=False).update_yaxes(
            visible=False)
    fig.show()

    # Question 3: Decision surface of best performing ensemble
    test_error = []
    num_classifiers_list = np.arange(1, n_learners)
    for T in num_classifiers_list:
        test_error.append(ada.partial_loss(test_X, test_y, T))
    T_hat = np.argmin(test_error) + 1
    go.Figure([decision_surface(lambda x: ada.partial_predict(x, T_hat) ,lims[0], lims[1], showscale=False),
               go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode='markers',
                          marker=dict(color=test_y, colorscale=custom))],
              layout=go.Layout(width=600, height=600, title=rf"$\text{{ensemble size achieved the lowest test error - }}iterations={T_hat},accuracy={1-test_error[T_hat]}$")).show()
    # Question 4: Decision surface with weighted samples
    D = ada.D_ / np.max(ada.D_)*10
    go.Figure([decision_surface(lambda x: ada.partial_predict(x, n_learners), lims[0], lims[1], showscale=False),
               go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode='markers', showlegend=False,
                          marker=dict(color=train_y, colorscale=class_colors(2),
                                      size=D))]).update_layout(title="Training set with a point size proportion", width=600, height=600).show()



if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0, n_learners=250, train_size=5000, test_size=500)
    fit_and_evaluate_adaboost(0.4, n_learners=250, train_size=5000, test_size=500)
