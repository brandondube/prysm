"""Some differentiable operators."""
from prysm.mathops import np

class SpatialGradient2D:
    """Spatial parital derivatives and backpropagation."""

    def forward_x(self, x):
        """Compute the X spatial gradient of an array."""
        assert x.ndim == 2, 'This operator only works on 2D arrays.'
        end = x.shape[1]
        ind_compute = slice(1, end-1)
        ind_lookahead = slice(2, end)
        out = np.zeros_like(x)
        out[:, ind_compute] = x[:, ind_lookahead] - x[:, ind_compute]
        return out

    def backprop_x(self, xbar):
        """Backpropagate through X spatial gradient of an array."""
        assert xbar.ndim == 2, 'This operator only works on 2D arrays.'
        end = xbar.shape[1]
        ind_compute = slice(1, end-1)
        ind_lookbehind = slice(0, end-2)
        out = np.zeros_like(xbar)
        out[:, ind_compute] = xbar[:, ind_lookbehind] - xbar[:, ind_compute]
        return out

    def forward_y(self, x):
        """Compute the Y spatial gradient of an array."""
        assert x.ndim == 2, 'This operator only works on 2D arrays.'
        end = x.shape[1]
        ind_compute = slice(1, end-1)
        ind_lookahead = slice(2, end)
        out = np.zeros_like(x)
        out[ind_compute, :] = x[ind_lookahead, :] - x[ind_compute, :]
        return out

    def reverse_y(self, xbar):
        """Backpropagate through Y spatial gradient of an array."""
        assert xbar.ndim == 2, 'This operator only works on 2D arrays.'
        end = xbar.shape[1]
        ind_compute = slice(1, end-1)
        ind_lookbehind = slice(0, end-2)
        out = np.zeros_like(xbar)
        out[ind_compute, :] = xbar[ind_lookbehind, :] - xbar[ind_compute, :]
        return out
