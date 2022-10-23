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


def calculate_loss_logistic(y, tx, w):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a non-negative loss

    1.52429481
    """
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]

    # ***************************************************
    # INSERT YOUR CODE HERE
    # ***************************************************
    return np.sum(
        -(1 / y.shape[0])
        * (
            y.T.dot(np.log(sigmoid(tx.dot(w))))
            + (1 - y).T.dot(np.log(1 - sigmoid(tx.dot(w))))
        )
    )


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


def calculate_gradient_logistic(y, tx, w):
    """compute the gradient of loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a vector of shape (D, 1)
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # ***************************************************
    return (1 / y.shape[0] * (sigmoid(tx.dot(w)) - y).T.dot(tx)).T


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


def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array

    """
    return 1 / (1 + np.exp(-t))


def calculate_hessian(y, tx, w):
    """return the Hessian of the loss function.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a hessian matrix of shape=(D, D)

    array([[0.28961235, 0.3861498 , 0.48268724],
           [0.3861498 , 0.62182124, 0.85749269],
           [0.48268724, 0.85749269, 1.23229813]])
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # ***************************************************
    S = sigmoid(tx.dot(w)).dot((1 - sigmoid(tx.dot(w))).T) * np.eye(
        tx.shape[0], tx.shape[0]
    )
    return 1 / tx.shape[0] * tx.T.dot(S).dot(tx)


def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression. Return the loss and the updated w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        gamma: float

    Returns:
        loss: scalar number
        w: shape=(D, 1)

    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # ***************************************************
    loss = calculate_loss_logistic(y, tx, w)
    w = w - gamma * (calculate_gradient_logistic(y, tx, w))
    return loss, w


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """return the loss, gradient of the loss, and hessian of the loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        loss: scalar number
        gradient: shape=(D, 1)
        hessian: shape=(D, D)
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # ***************************************************
    w = initial_w
    w = w.T
    loss = calculate_loss_logistic(y, tx, w)
    for n_iter in range(max_iters):
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
    return w, loss


def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss and gradient.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        lambda_: scalar

    Returns:
        loss: scalar number
        gradient: shape=(D, 1)
    """
    # ***************************************************
    # ***************************************************
    loss = calculate_loss_logistic(y, tx, w) + lambda_ * np.sum(np.square(w))
    grad = calculate_gradient_logistic(y, tx, w) + 2 * lambda_ * w
    return loss, grad


def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        gamma: scalar
        lambda_: scalar

    Returns:
        loss: scalar number
        w: shape=(D, 1)

    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # ***************************************************
    loss, grad = penalized_logistic_regression(y, tx, w, lambda_)
    # ***************************************************
    # INSERT YOUR CODE HERE
    # ***************************************************
    w = w - gamma * grad
    return loss, w


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """return the loss and gradient.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        lambda_: scalar

    Returns:
        loss: scalar number
        gradient: shape=(D, 1)

    array([[-0.08370763],
           [ 0.2467104 ],
           [ 0.57712843]])
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # ***************************************************
    w = initial_w
    w = w.T
    loss = calculate_loss_logistic(y, tx, w)
    for iter in range(max_iters):
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        if iter == max_iters - 2:
            w_1 = w
    return w, loss - lambda_ * np.sum(np.square(w_1))
