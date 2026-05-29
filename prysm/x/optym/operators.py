"""Some differentiable operators."""
from prysm.mathops import np


class SpatialGradient2D:
    """Spatial parital derivatives and adjoints."""

    def forward_x(self, x):
        """Compute the X spatial gradient of an array."""
        assert x.ndim == 2, 'This operator only works on 2D arrays.'
        end = x.shape[1]
        ind_compute = slice(1, end-1)
        ind_lookahead = slice(2, end)
        out = np.zeros_like(x)
        out[:, ind_compute] = x[:, ind_lookahead] - x[:, ind_compute]
        return out

    def adjoint_x(self, xbar):
        """Backpropagate through X spatial gradient of an array."""
        assert xbar.ndim == 2, 'This operator only works on 2D arrays.'
        end = xbar.shape[1]
        ind_compute = slice(1, end-1)
        ind_lookahead = slice(2, end)
        out = np.zeros_like(xbar)
        out[:, ind_compute] -= xbar[:, ind_compute]
        out[:, ind_lookahead] += xbar[:, ind_compute]
        return out

    def forward_y(self, x):
        """Compute the Y spatial gradient of an array."""
        assert x.ndim == 2, 'This operator only works on 2D arrays.'
        end = x.shape[0]
        ind_compute = slice(1, end-1)
        ind_lookahead = slice(2, end)
        out = np.zeros_like(x)
        out[ind_compute, :] = x[ind_lookahead, :] - x[ind_compute, :]
        return out

    def adjoint_y(self, xbar):
        """Backpropagate through Y spatial gradient of an array."""
        assert xbar.ndim == 2, 'This operator only works on 2D arrays.'
        end = xbar.shape[0]
        ind_compute = slice(1, end-1)
        ind_lookahead = slice(2, end)
        out = np.zeros_like(xbar)
        out[ind_compute, :] -= xbar[ind_compute, :]
        out[ind_lookahead, :] += xbar[ind_compute, :]
        return out
