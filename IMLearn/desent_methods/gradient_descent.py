from __future__ import annotations
from typing import Callable, NoReturn
import numpy as np

from IMLearn.base import BaseModule, BaseLR
from .learning_rate import FixedLR

OUTPUT_VECTOR_TYPE = ["last", "best", "average"]


def default_callback(**kwargs) -> NoReturn:
    pass


class GradientDescent:
    """
    Gradient Descent algorithm

    Attributes:
    -----------
    learning_rate_: BaseLR
        Learning rate strategy for retrieving the learning rate at each iteration t of the algorithm

    tol_: float
        The stopping criterion. Training stops when the Euclidean norm of w^(t)-w^(t-1) is less than
        specified tolerance

    max_iter_: int
        The maximum number of GD iterations to be performed before stopping training

    out_type_: str
        Type of returned solution:
            - `last`: returns the point reached at the last GD iteration
            - `best`: returns the point achieving the lowest objective
            - `average`: returns the average point over the GD iterations

    callback_: Callable[[...], None], default=default_callback
        A callable function to be called after each update of the model while fitting to given data.
        Callable function receives as input any argument relevant for the current GD iteration. Arguments
        are specified in the `GradientDescent.fit` function
    """
    def __init__(self,
                 learning_rate: BaseLR = FixedLR(1e-3),
                 tol: float = 1e-5,
                 max_iter: int = 1000,
                 out_type: str = "last",
                 callback: Callable[[GradientDescent, ...], None] = default_callback):
        """
        Instantiate a new instance of the GradientDescent class

        Parameters
        ----------
        learning_rate: BaseLR, default=FixedLR(1e-3)
            Learning rate strategy for retrieving the learning rate at each iteration t of the algorithm

        tol: float, default=1e-5
            The stopping criterion. Training stops when the Euclidean norm of w^(t)-w^(t-1) is less than
            specified tolerance

        max_iter: int, default=1000
            The maximum number of GD iterations to be performed before stopping training

        out_type: str, default="last"
            Type of returned solution. Supported types are specified in class attributes

        callback: Callable[[...], None], default=default_callback
            A callable function to be called after each update of the model while fitting to given data.
            Callable function receives as input any argument relevant for the current GD iteration. Arguments
            are specified in the `GradientDescent.fit` function
        """
        self.learning_rate_ = learning_rate
        if out_type not in OUTPUT_VECTOR_TYPE:
            raise ValueError("output_type not supported")
        self.out_type_ = out_type
        self.tol_ = tol
        self.max_iter_ = max_iter
        self.callback_ = callback

    def fit(self, f: BaseModule, X: np.ndarray, y: np.ndarray):
        """
        Optimize module using Gradient Descent iterations over given input samples and responses

        Parameters
        ----------
        f : BaseModule
            Module of objective to optimize using GD iterations
        X : ndarray of shape (n_samples, n_features)
            Input data to optimize module over
        y : ndarray of shape (n_samples, )
            Responses of input data to optimize module over

        Returns
        -------
        solution: ndarray of shape (n_features)
            Obtained solution for module optimization, according to the specified self.out_type_

        Notes
        -----
        - Optimization is performed as long as self.max_iter_ has not been reached and that
        Euclidean norm of w^(t)-w^(t-1) is more than the specified self.tol_

        - At each iteration the learning rate is specified according to self.learning_rate_.lr_step

        - At the end of each iteration the self.callback_ function is called passing self and the
        following named arguments:
            - solver: GradientDescent
                self, the current instance of GradientDescent
            - weights: ndarray of shape specified by module's weights
                Current weights of objective
            - val: ndarray of shape specified by module's compute_output function
                Value of objective function at current point, over given data X, y
            - grad:  ndarray of shape specified by module's compute_jacobian function
                Module's jacobian with respect to the weights and at current point, over given data X,y
            - t: int
                Current GD iteration
            - eta: float
                Learning rate used at current iteration
            - delta: float
                Euclidean norm of w^(t)-w^(t-1)

        """
        # best = f.weights
        # best_val = f.compute_output(X=X, y=y)
        # sum = np.ndarray(f.shape, dtype=float)
        # for i in range(self.max_iter_):
        #     grad, eta = f.compute_jacobian(X=X, y=y), self.learning_rate_.lr_step(t=i+1)
        #     new_weights = f.weights - eta*grad
        #     delta = np.linalg.norm(new_weights-f.weights_)
        #     sum += new_weights
        #     f.weights = new_weights
        #     val = f.compute_output(X=X, y=y)
        #     self.callback_(solver=self, weights=new_weights, val=f.compute_output(X, y), grad=f.compute_jacobian(X,y), t=i, eta=eta, delta=delta)
        #     if val < best_val :
        #         best = f.weights
        #         best_val = val
        #     if delta < self.tol_:
        #         break
        # if self.out_type_ == 'last':
        #     return f.weights
        # if self.out_type_ == 'best':
        #     return best
        # return 1/(i+1) * sum



        solution = f.weights_  # start solution
        num_iters = 0
        delta = np.linalg.norm(solution)
        best_value = f.compute_output(X=X, y=y)
        best_solution = solution
        total = 0

        while num_iters < self.max_iter_ and delta > self.tol_:
            eta = self.learning_rate_.lr_step(t=num_iters)
            grad = f.compute_jacobian(X=X, y=y)
            cur_solution = solution - eta * grad
            delta = np.linalg.norm(cur_solution - solution)
            cur_val = f.compute_output(X=X, y=y)
            if cur_val < best_value:
                best_solution = cur_solution
                best_value = cur_val
            total += cur_solution
            solution = cur_solution
            num_iters += 1
            f.weights = cur_solution
            self.callback_(weights=f.weights, val=f.compute_output(X=X, y=y))

        if self.out_type_ == 'last':
            return solution
        if self.out_type_ == 'best':
            return best_solution
        return total / (num_iters + 1)





