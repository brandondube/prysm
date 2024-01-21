"""Cost functions, aka figures of merit for models."""
import numpy as np


class BiasAndGainInvariantError:
    """Bias and gain invariant error.

    This cost function computes internal least mean squares estimates of the
    overall bias (DC pedestal) and gain between the signal I and D.  This
    objective is useful when the overall signal level is ambiguous in phase
    retrieval type problems, and can significantly help stabilize the
    optimization process.
    """
    def __init__(self):
        """Create a new BiasAndGainInvariantError instance."""
        self.R = None
        self.alpha = None
        self.beta = None
        self.I = None  # NOQA
        self.D = None
        self.mask = None

    def forward(self, I, D, mask):  # NOQA
        """Forward cost evaluation.

        Parameters
        ----------
        I : numpy.ndarray
            'intensity' or model data, any float dtype, any shape
        D : numpy.ndarray
            'data' or true mesaurement to be matched, any float dtype, any shape
        mask : numpy.ndarray
            logical array with elements to keep (True) or exclude (False)

        Returns
        -------
        float
            scalar cost

        """
        # intermediate variables
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
        self.R = R
        self.alpha = alpha
        self.beta = beta
        return err

    def backprop(self):
        """Returns the first step of gradient backpropagation, an array of the same shape as I."""
        R = self.R
        alpha = self.alpha
        beta = self.beta
        I = self.I  # NOQA
        D = self.D
        mask = self.mask

        out = np.zeros_like(I)
        I = I[mask]  # NOQA
        D = D[mask]
        out[mask] = 2*R*alpha*((alpha*I + beta) - D)
        return out


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
