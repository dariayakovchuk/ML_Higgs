import numpy as np


def compute_loss(y, tx, w):
    """Calculate the loss using either MSE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    N = y.shape[0]
    print(np.dot(tx, w))
    cost = (1 / (2 * N)) * np.sum((y - np.dot(tx, w)) ** 2)
    return cost


def least_squares(y, tx):
    """
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.
    """
    N = y.shape[0]
    w = np.linalg.solve(np.dot(tx.T, tx), np.dot(tx.T, y))
    cost = compute_loss(y, tx, w)
    return w, cost


def ridge_regression(y, tx, lambda_):
    """
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
    """
    n = tx.shape[1]
    I = np.eye(n)
    w_ridge = np.linalg.solve(
        np.dot(tx.T, tx) + 2 * lambda_ * tx.shape[0] * I, np.dot(tx.T, y)
    )
    loss = compute_loss(y, tx, w_ridge)
    return w_ridge, loss


def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        An numpy array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    return (-1 / np.shape(tx)[0] * np.dot(tx.T, (y - np.dot(tx, w)))).flatten()


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of GD
    """
    w = initial_w
    w = w.T
    loss = compute_loss(y, tx, w.T)
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w.T)
        w = w - gamma * gradient
        loss = compute_loss(y, tx, w.T)
    # w = w.reshape(w.shape[0], w.shape[1])
    return w.T, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD
    """
    w = initial_w
    w = w.T
    loss = compute_loss(y, tx, w.T)
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w.T)
        w = w - gamma * gradient
        loss = compute_loss(y, tx, w.T)
    # w = w.reshape(w.shape[0], w.shape[1])
    return w.T, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """

    :param y:
    :param tx:
    :param initial_w:
    :param max_iters:
    :param gamma:
    :return:
    """
    return 0


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """

    :param y:
    :param tx:
    :param lambda_:
    :param initial_w:
    :param max_iters:
    :param gamma:
    :return:
    """
    return 0


gamma = 0.1
max_iters = 2
initial_w = np.array([[0.5], [1.0]])
y = np.array([[0.1], [0.3], [0.5]])
tx = np.array([[2.3, 3.2], [1.0, 0.1], [1.4, 2.3]])
a, m = ridge_regression(y, tx, 1)
print(a, m)
