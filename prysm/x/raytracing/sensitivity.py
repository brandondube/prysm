"""Jacobians of merit functions w.r.t. prescription parameters.

Two backends are exposed via the ``method`` kwarg of :func:`merit_jacobian`:

- ``'fd'`` (default): central finite differences.  Always works regardless
  of which numerical backend ``prysm.mathops`` is pointing at.
- ``'autograd'``: PyTorch reverse-mode automatic differentiation.  Requires
  the prysm backend to be torch (``prysm.mathops.set_backend_to_pytorch()``)
  so that the surface sag_and_normal callbacks and the merit function are torch-traced
  end-to-end.

Parameters are addressed via (getter, setter) callable pairs.  The helper
:func:`surface_param` constructs one from a ``(surface_index, param_name)``
tuple, which is convenient for the common case of perturbing a stored value
in ``Surface.params``.

"""

from prysm.conf import config
from prysm.mathops import np


def surface_param(prescription, surface_index, param_name):
    """Build a ``(getter, setter)`` pair targeting one entry of a surface's
    ``params`` dict.

    Mutating the entry through ``setter`` is visible to the sag_and_normal closure on
    that surface (which reads ``params[name]`` on every call), so subsequent
    raytraces will see the perturbed value.

    """
    params = prescription[surface_index].params
    if params is None or param_name not in params:
        raise KeyError(
            f'surface {surface_index} has no parameter {param_name!r} '
            '(check Surface.params)'
        )

    def getter():
        return params[param_name]

    def setter(v):
        params[param_name] = v

    return getter, setter


def vertex_z_param(prescription, surface_index):
    """``(getter, setter)`` for the z coordinate of a surface's vertex.

    Useful for perturbing surface spacings.

    """
    P = prescription[surface_index].P

    def getter():
        return float(P[2])

    def setter(v):
        P[2] = v

    return getter, setter


def _jacobian_fd(prescription, parameters, merit, step, merit_kwargs):
    n_p = len(parameters)
    J = np.empty(n_p, dtype=config.precision)
    for i, (g, s) in enumerate(parameters):
        v0 = float(g())
        h = step * (abs(v0) if v0 != 0.0 else 1.0)
        s(v0 + h)
        fp = float(merit(prescription, **merit_kwargs))
        s(v0 - h)
        fm = float(merit(prescription, **merit_kwargs))
        s(v0)
        J[i] = (fp - fm) / (2.0 * h)
    return J


def _jacobian_autograd(prescription, parameters, merit, merit_kwargs):
    if np.__name__ != 'torch':
        raise RuntimeError(
            "method='autograd' requires the prysm backend to be torch.  "
            "Call prysm.mathops.set_backend_to_pytorch() before invoking."
        )
    import torch  # noqa: F401 — required so that np.__name__ check is meaningful
    originals = [g() for g, _ in parameters]
    leaves = [
        # autograd path requires torch backend; `np.float64` here resolves
        # to `torch.float64` via the mathops shim.  We deliberately don't
        # use `config.precision` because that is captured as a numpy dtype
        # at config-init time and torch.tensor's dtype kwarg requires a
        # torch.dtype, not a numpy one.
        np.tensor(float(v), dtype=np.float64, requires_grad=True)
        for v in originals
    ]
    for (g, s), leaf in zip(parameters, leaves):
        s(leaf)
    try:
        loss = merit(prescription, **merit_kwargs)
        loss.backward()
        J = np.empty(len(parameters), dtype=config.precision)
        for i, leaf in enumerate(leaves):
            grad = leaf.grad
            J[i] = 0.0 if grad is None else float(grad)
    finally:
        for (g, s), v in zip(parameters, originals):
            s(v)
    return J


def merit_jacobian(prescription, parameters, merit,
                   method='fd', step=1e-6, **merit_kwargs):
    """Jacobian of ``merit(prescription, **merit_kwargs)`` w.r.t. each
    parameter.

    Parameters
    ----------
    prescription : sequence of Surface
        the prescription in its current (nominal) state.  In FD mode this
        sequence is mutated transiently and restored before return.  In
        autograd mode it is mutated to leaf tensors and restored similarly.
    parameters : sequence of (getter, setter)
        callable pairs targeting each scalar parameter.  Use
        :func:`surface_param` or :func:`vertex_z_param` to build the common
        cases.
    merit : callable
        ``merit(prescription, **merit_kwargs) -> scalar`` (Python float for
        ``method='fd'``; torch scalar tensor for ``method='autograd'``).
    method : {'fd', 'autograd'}, optional
        differentiation backend.  Default ``'fd'``.
    step : float, optional
        FD step size, scaled by ``|v0|`` (or 1 if v0 == 0) per parameter.
        Default 1e-6 — appropriate for fp64 finite differences on smooth
        merits.  Only used by ``method='fd'``.
    **merit_kwargs
        forwarded to ``merit`` on every evaluation.

    Returns
    -------
    J : ndarray
        shape ``(len(parameters),)`` — gradient of the scalar merit w.r.t.
        each parameter, in the order the parameters were supplied.

    """
    if method == 'fd':
        return _jacobian_fd(prescription, parameters, merit, step, merit_kwargs)
    if method == 'autograd':
        return _jacobian_autograd(prescription, parameters, merit, merit_kwargs)
    raise ValueError(f"method must be 'fd' or 'autograd', got {method!r}")
