"""Jacobian of a scalar merit w.r.t. a LensData's dense free vector.

Two backends via the ``method`` kwarg of :func:`merit_jacobian_free`:

- ``'fd'`` (default): central finite differences over the free vector.  Works
  regardless of which numerical backend ``prysm.mathops`` points at.
- ``'autograd'``: PyTorch reverse-mode autodiff through ``update`` ->
  ``to_surfaces`` -> merit.  Requires the prysm backend to be torch
  (``prysm.mathops.set_backend_to_pytorch()``).

"""

from prysm.conf import config
from prysm.mathops import np


def merit_jacobian_free(lensdata, merit, method='fd', step=1e-6):
    """Gradient of a scalar merit w.r.t. a LensData's dense free vector.

    The merit is a zero-argument callable returning the scalar figure of merit
    of the LensData in its current state.  ``method='fd'`` perturbs each free
    DOF with central differences (mutating the free vector in place and
    restoring it on return).  ``method='autograd'`` differentiates through
    ``update`` -> ``to_surfaces`` -> merit, and requires the prysm backend to
    be torch.

    Parameters
    ----------
    lensdata : LensData
        system whose free vector is differentiated.  Restored to its nominal
        free vector before return.
    merit : callable
        ``merit() -> scalar`` evaluating the current LensData state.
    method : {'fd', 'autograd'}, optional
        differentiation backend.  Default ``'fd'``.
    step : float, optional
        FD step, scaled by ``|x_i|`` (or 1 if zero) per DOF.  Default 1e-6.

    Returns
    -------
    J : ndarray
        shape ``(n_free,)`` gradient of the merit w.r.t. the free vector.

    """
    x0 = lensdata.pack()
    n = len(x0)
    if method == 'fd':
        J = np.empty(n, dtype=config.precision)
        try:
            for i in range(n):
                v0 = float(x0[i])
                h = step * (abs(v0) if v0 != 0.0 else 1.0)
                x = np.array(x0, copy=True)
                x[i] = v0 + h
                lensdata.update(x)
                fp = float(merit())
                x[i] = v0 - h
                lensdata.update(x)
                fm = float(merit())
                J[i] = (fp - fm) / (2.0 * h)
        finally:
            lensdata.update(x0)
        return J
    if method == 'autograd':
        if np.__name__ != 'torch':
            raise RuntimeError(
                "method='autograd' requires the prysm backend to be torch.  "
                "Call prysm.mathops.set_backend_to_pytorch() before invoking."
            )
        leaf = np.tensor(np.array(x0, dtype=np.float64), requires_grad=True)
        try:
            lensdata.update(leaf)
            loss = merit()
            loss.backward()
            grad = leaf.grad
            J = (np.zeros(n, dtype=config.precision)
                 if grad is None else np.array(grad))
        finally:
            lensdata.update(x0)
        return J
    raise ValueError(f"method must be 'fd' or 'autograd', got {method!r}")
