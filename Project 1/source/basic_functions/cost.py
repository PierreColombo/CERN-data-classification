
def calculate_ridge_loss(y, tx, w):
    """compute the loss by mse."""
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse


def calculate_loss(y, tx, w, lambda_):
    """
    Calculates loss for regularized logistic regression
    :param y: output (N, 1)
    :param tx: input variables (N, D)
    :param w: weights vector w (D, 1)
    :return: loss
    """
    n = tx.shape[0]
    d = tx.shape[1]

    pred = np.dot(tx, w)
    return 1 / n * (np.dot(np.log(1 + np.exp(pred)) - y * pred, np.ones(n)) + lambda_ / 2.0 * np.dot(w[1:], w[1:]))
