"""Scalar-merit Jacobian over a LensData free vector."""

from prysm.conf import config
from prysm.mathops import np


def central_difference(probe, base, h):
    """Central-difference probe of a scalar about base by +/- h.

    Returns (probe(base + h), probe(base - h)).
    """
    return float(probe(base + h)), float(probe(base - h))


def fd_jacobian(f, x, step=1e-6, mask=None):
    """Central-difference gradient of a scalar f over the vector x.

    Parameters
    ----------
    f : callable
        f(x) -> scalar over the free vector.
    x : array_like
        the point at which the gradient is taken.
    step : float, optional
        relative FD half-step, scaled by abs(x_i) or 1.
    mask : array_like of bool, optional
        components to differentiate; others keep derivative 0.

    Returns
    -------
    J : ndarray
        shape (len(x),) gradient of f at x.

    """
    x = np.asarray(x)
    n = len(x)
    J = np.zeros(n, dtype=config.precision)
    for i in range(n):
        if mask is not None and not mask[i]:
            continue
        v0 = float(x[i])
        h = step * (abs(v0) if v0 != 0.0 else 1.0)

        def probe(value, i=i):
            xx = np.array(x, copy=True)
            xx[i] = value
            return f(xx)

        fp, fm = central_difference(probe, v0, h)
        J[i] = (fp - fm) / (2.0 * h)
    return J


def merit_jacobian_free(dofs, merit, method='fd', step=1e-6):
    """Gradient of a scalar merit w.r.t. a system's dense free vector.

    Parameters
    ----------
    dofs : DesignState
        free-vector owner (pack/update) whose DOFs are differentiated.
        Restored to its nominal free vector before return.
    merit : callable
        `merit() -> scalar` evaluating the current system state.
    method : {'fd', 'autograd'}, optional
        differentiation backend.  `'autograd'` requires the torch backend.
    step : float, optional
        FD step, scaled by abs(x_i) (or 1 if zero) per DOF.  Default 1e-6.

    Returns
    -------
    J : ndarray
        shape `(n_free,)` gradient of the merit w.r.t. the free vector.

    """
    x0 = dofs.pack()
    n = len(x0)
    if method == 'fd':
        def f(x):
            dofs.update(x)
            return float(merit())
        try:
            return fd_jacobian(f, x0, step=step)
        finally:
            dofs.update(x0)
    if method == 'autograd':
        if np.__name__ != 'torch':
            raise RuntimeError(
                "method='autograd' requires the prysm backend to be torch.  "
                "Call prysm.mathops.set_backend_to_pytorch() before invoking."
            )
        leaf = np.tensor(np.array(x0, dtype=np.float64), requires_grad=True)
        try:
            dofs.update(leaf)
            loss = merit()
            loss.backward()
            grad = leaf.grad
            J = (np.zeros(n, dtype=config.precision)
                 if grad is None else np.array(grad))
        finally:
            dofs.update(x0)
        return J
    raise ValueError(f"method must be 'fd' or 'autograd', got {method!r}")
