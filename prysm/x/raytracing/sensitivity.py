"""Jacobian of a scalar merit w.r.t. a LensData's dense free vector.

Two backends via the `method` kwarg of merit_jacobian_free:

- `'fd'` (default): central finite differences over the free vector.  Works
  regardless of which numerical backend `prysm.mathops` points at.
- `'autograd'`: PyTorch reverse-mode autodiff through `update` ->
  `to_surfaces` -> merit.  Requires the prysm backend to be torch
  (`prysm.mathops.set_backend_to_pytorch()`).

"""

from prysm.conf import config
from prysm.mathops import np


def central_difference(probe, base, h):
    """Central-difference probe of a scalar about base by +/- h.

    probe(value) -> scalar evaluates the figure of merit with the single varied
    DOF set to value; base is its unperturbed value and h the half-step.
    Returns (f_plus, f_minus) = (probe(base + h), probe(base - h)); the
    derivative is (f_plus - f_minus) / (2 * h).  A caller that also needs the
    raw evaluations -- a sensitivity table reporting merit_plus / merit_minus --
    reads them off the pair.  probe leaves the DOF at the last value it set, so
    the caller restores it.
    """
    return float(probe(base + h)), float(probe(base - h))


def fd_jacobian(f, x, step=1e-6, mask=None):
    """Central-difference gradient of a scalar f over the vector x.

    f(x) -> scalar.  Component i is stepped by h_i = step * (|x_i| or 1); a
    False entry in mask leaves that component's derivative at 0.  x is copied
    for each probe (never mutated in place here), so f sees exactly one varied
    component at a time.  Not optym's forward-difference vector-residual
    Jacobian; this is a scalar-merit gradient.

    Parameters
    ----------
    f : callable
        f(x) -> scalar over the free vector.
    x : array_like
        the point at which the gradient is taken.
    step : float, optional
        relative FD step; the per-component half-step is step * (|x_i| or 1).
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

    The merit is a zero-argument callable returning the scalar figure of merit
    of the system in its current state.  `method='fd'` perturbs each free
    DOF with central differences (mutating the free vector in place and
    restoring it on return).  `method='autograd'` differentiates through
    `update` -> `to_surfaces` -> merit, and requires the prysm backend to
    be torch.

    Parameters
    ----------
    dofs : DesignState
        free-vector owner (pack/update) whose DOFs are differentiated.
        Restored to its nominal free vector before return.
    merit : callable
        `merit() -> scalar` evaluating the current system state.
    method : {'fd', 'autograd'}, optional
        differentiation backend.  Default `'fd'`.
    step : float, optional
        FD step, scaled by `|x_i|` (or 1 if zero) per DOF.  Default 1e-6.

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
