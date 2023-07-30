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


class LogLikelyhood:
    def __init__(self):
        pass

    def forward(self, I, D, mask):
        # I, D, mask for symmetry to bias and gain invariant
        # internally use y, yhat because that's how the this is usually written
        y = D
        yhat = I

        ylogyhat = y*np.log(yhat)
        one_minus_y = 1 - y
        log_one_minus_yhat = np.log(1-yhat)
        cost = -(ylogyhat + one_minus_y*log_one_minus_yhat).sum()
        return cost

    def backprop(self):
        pass
