"""Cost functions, aka figures of merit for models."""
import numpy as np


class BiasAndGainInvariantError:
    def __init__(self):
        self.R = None
        self.alpha = None
        self.beta = None
        self.I = None  # NOQA
        self.D = None
        self.mask = None

    def forward(self, I, D, mask):  # NOQA
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
        """Returns Ibar."""
        R = self.R
        alpha = self.alpha
        beta = self.beta
        I = self.I  # NOQA
        D = self.D
        mask = self.mask

        out = np.zeros_like(I)
        I = I[mask]
        D = D[mask]
        out[mask] = 2*R*alpha*((alpha*I + beta) - D)
        return out
