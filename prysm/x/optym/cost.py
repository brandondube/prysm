"""Cost functions, aka figures of merit for models."""
from prysm.mathops import np


def bias_and_gain_invariant_error(I, D, mask=None):
    """Bias and gain invariant variant of mean square error.

    Parameters
    ----------
    I : numpy.ndarray
        "model data"
    D : numpy.ndarray
        "truth data"
    mask : numpy.ndarray, optional
        True where M should contribute to the cost, False where it should not

    Returns
    -------
    float, numpy.ndarray
        cost, dcost/dM

    """
    I = I[mask]  # NOQA
    D = D[mask]
    Ihat = I - I.mean()  # zero mean
    Dhat = D - D.mean()

    N = I.size

    num = (Ihat*Dhat).sum()
    den = (Ihat*Ihat).sum()
    alpha = num/den

    alphaI = alpha*I

    beta = (D-alphaI)/N

    R = 1/((D*D).sum())
    raw_err = (alphaI + beta) - D
    err = R*(raw_err*raw_err).sum()

    grad = np.zeros_like(I)
    grad[mask] = 2*R*alpha*raw_err
    return err, grad


def mean_square_error(M, D, mask=None):
    """Mean square error.

    Parameters
    ----------
    M : numpy.ndarray
        "model data"
    D : numpy.ndarray
        "truth data"
    mask : numpy.ndarray, optional
        True where M should contribute to the cost, False where it should not

    Returns
    -------
    float, numpy.ndarray
        cost, dcost/dM

    """
    diff = (M-D)
    if mask is not None:
        diff = diff[mask]

    alpha = 1 / diff.size
    cost = (diff*diff).sum() * alpha

    # backprop
    if mask is not None:
        grad = np.zeros_like(M)
        grad[mask] = 2 * alpha * diff
    else:
        grad = 2 * alpha * diff

    return cost, grad


def negative_loglikelihood(y, yhat, mask=None):
    """Negative log likelihood.

    Parameters
    ----------
    y : numpy.ndarray
        predicted values; typically the output of a model
    yhat : numpy.ndarray
        truth or target values
    mask : numpy.ndarray, optional
        True where M should contribute to the cost, False where it should not

    Returns
    -------
    float, numpy.ndarray
        cost, dcost/dy

    """
    if mask is not None:
        y = y[mask]
        yhat = yhat[mask]

    sub1 = 1-y
    sub2 = 1-yhat
    prefix = 1/y.size  #               1-yhat        1-y  # NOQA flake8 doesn't like comment starting with space
    cost = -prefix * (yhat*np.log(y) + (sub2)*np.log(sub1)).sum()

    #                   1-yhat  1-y
    dcost = (-yhat/y) + (sub2)/(sub1)
    dcost *= prefix

    if mask is not None:
        dcost2 = np.zeros(mask.shape, dtype=y.dtype)
        dcost2[mask] = dcost
        dcost = dcost2

    return cost, dcost
