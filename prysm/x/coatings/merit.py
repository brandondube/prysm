"""Merit terms (objectives) for coating design.

A merit term ties a physical response quantity (reflectance, transmittance,
per-layer absorptance, internal field) over a grid of (wavelength, angle,
polarization) samples to a target, with a weight.  Each term knows how to
evaluate its weighted sum-of-squares value, its residual vector (for the
least-squares path), and -- through the analytic adjoint in diff.py -- its
gradient with respect to every layer thickness.

A MeritFunction is a weighted collection of terms; its value and gradient are
the sums of the terms'.  This is the object refine / synthesize optimize.

Conventions
-----------
Wavelengths and thicknesses are microns; angles are radians (the low-level
convention -- use numpy.radians on degrees).  Polarization is 's', 'p', or
'avg' (unpolarized = mean of s and p).  Targets and weights broadcast against
the (wavelength, angle) sample grid.
"""

from prysm.conf import config
from prysm.mathops import np

from .diff import forward_eval, thickness_gradient, assembly_cotangent


def _as_grid(x):
    arr = np.asarray(x, dtype=config.precision)
    return arr


def _validate_term_shapes(wvl, theta, target, weight):
    if wvl.ndim == 1 and theta.ndim == 1 and wvl.size > 1 and theta.size > 1:
        raise ValueError(
            'wvl and theta are both 1-D; pass meshgridded arrays for a '
            'spectral/angular grid')
    try:
        np.broadcast_arrays(wvl, theta, target, weight)
    except ValueError as exc:
        raise ValueError(
            'wvl, theta, target, and weight must be broadcast-compatible'
        ) from exc


class _Term:
    """Base spectral/angular merit term over a (wvl, theta, pol) grid.

    Subclasses define the physical quantity and how its cotangent enters the
    thickness gradient.
    """

    quantity = None

    def __init__(self, wvl, theta=0.0, pol='avg', target=0.0, weight=1.0):
        """Define a merit term over a (wvl, theta, pol) sample grid.

        Parameters
        ----------
        wvl : float or ndarray
            wavelength sample(s), microns.
        theta : float or ndarray, optional
            angle(s) of incidence in the ambient medium, radians.
        pol : str, optional
            polarization the quantity is evaluated for, 's', 'p', or 'avg'
            (the unpolarized mean of s and p).
        target : float or ndarray, optional
            desired value of the quantity at each sample.
        weight : float or ndarray, optional
            relative weight of each sample in the sum-of-squares merit.

        wvl, theta, target, and weight must broadcast to a common grid shape.

        """
        self.wvl = _as_grid(wvl)
        self.theta = _as_grid(theta)
        pol = pol.lower()
        if pol not in ('s', 'p', 'avg'):
            raise ValueError("pol must be 's', 'p', or 'avg'")
        self.pol = pol
        self.target = _as_grid(target)
        self.weight = _as_grid(weight)
        _validate_term_shapes(self.wvl, self.theta, self.target, self.weight)

    # --- subclass hooks ---------------------------------------------------
    def _quantity(self, fwd):
        raise NotImplementedError

    def _seed_kw(self, fwd, dq):
        """Map a cotangent dq on this term's quantity to diff-engine seed kwargs.

        Returns a dict like {'dR': ...} / {'dA': ...} suitable for
        diff.thickness_gradient / index_gradient / assembly_cotangent.
        """
        raise NotImplementedError

    def _is_assembly_quantity(self):
        """Whether the quantity depends on the stack only through assembled M.

        True for reflectance / transmittance (needle-eligible); False for the
        field / absorptance terms that read the internal partial products.
        """
        return False

    # --- shared machinery -------------------------------------------------
    def _pols(self):
        return ('s', 'p') if self.pol == 'avg' else (self.pol,)

    def _evaluate(self, stack):
        """Return the combined quantity q, the per-pol ForwardEvals, and q list."""
        pols = self._pols()
        fwds = [forward_eval(stack, self.wvl, self.theta, p) for p in pols]
        qs = [self._quantity(f) for f in fwds]
        q = qs[0]
        for extra in qs[1:]:
            q = q + extra
        q = q / len(qs)
        return q, fwds

    def residuals(self, stack):
        """Weighted residual vector sqrt(w) (q - target), flattened over samples."""
        q, _ = self._evaluate(stack)
        res = np.sqrt(self.weight) * (q - self.target)
        return np.atleast_1d(res).ravel()

    def value(self, stack):
        """Weighted sum of squared deviations from target (a scalar)."""
        q, _ = self._evaluate(stack)
        return float(np.sum(self.weight * (q - self.target) ** 2))

    def _dF_dq(self, q, npol):
        dF_dq = 2 * self.weight * (q - self.target)
        return np.broadcast_to(dF_dq, q.shape) / npol

    def value_and_grad(self, stack, grad_fn=thickness_gradient):
        """Scalar value and gradient via grad_fn (thickness_gradient default).

        Pass diff.index_gradient as grad_fn for index-variable (rugate) design.
        """
        q, fwds = self._evaluate(stack)
        val = float(np.sum(self.weight * (q - self.target) ** 2))
        dF_dq = self._dF_dq(q, len(fwds))
        grad = np.zeros(len(stack), dtype=config.precision)
        for f in fwds:
            grad = grad + grad_fn(f, **self._seed_kw(f, dF_dq))
        return val, grad

    def assembly_seeds(self, stack):
        """List of (ForwardEval, M_cotangent) pairs for the needle function.

        One pair per polarization sample; the assembled-matrix cotangent already
        carries this term's weight and the (q - target) residual, so the needle
        function contracts it directly against the inserted-layer perturbation.
        Only reflectance / transmittance terms qualify (assembly-only quantity).
        """
        if not self._is_assembly_quantity():
            raise NotImplementedError(
                'needle synthesis supports reflectance / transmittance targets')
        q, fwds = self._evaluate(stack)
        dF_dq = self._dF_dq(q, len(fwds))
        return [(f, assembly_cotangent(f, **self._seed_kw(f, dF_dq)))
                for f in fwds]


class Reflectance(_Term):
    """Target the intensity reflectance R = |r|^2 over a sample grid."""

    quantity = 'R'

    def _quantity(self, fwd):
        return fwd.R_value

    def _seed_kw(self, fwd, dq):
        return {'dR': dq}

    def _is_assembly_quantity(self):
        return True


class Transmittance(_Term):
    """Target the intensity transmittance T over a sample grid."""

    quantity = 'T'

    def _quantity(self, fwd):
        return fwd.T_value

    def _seed_kw(self, fwd, dq):
        return {'dT': dq}

    def _is_assembly_quantity(self):
        return True


class LayerAbsorptance(_Term):
    """Target the absorptance A of one named layer over a sample grid.

    layer is the index of the layer (ambient side first); the absorptance is the
    fraction of incident power dissipated within it.
    """

    quantity = 'A'

    def __init__(self, layer, wvl, theta=0.0, pol='avg', target=0.0, weight=1.0):
        super().__init__(wvl, theta=theta, pol=pol, target=target, weight=weight)
        self.layer = int(layer)

    def _quantity(self, fwd):
        return fwd.A_value[self.layer]

    def _seed_kw(self, fwd, dq):
        dA = np.zeros(fwd.A_value.shape, dtype=config.precision)
        dA[self.layer] = dq
        return {'dA': dA}


class FieldIntensityAtBoundary(_Term):
    """Target the standing-wave intensity |E|^2 at one boundary.

    Boundary k lies between layer k-1 and layer k (boundary 0 is the ambient /
    first-layer interface, boundary N is the last-layer / substrate interface).
    Minimizing the peak interface intensity is the laser-damage-resistance
    objective; maximizing it inside an active layer is the inverse.
    """

    quantity = 'Esq'

    def __init__(self, boundary, wvl, theta=0.0, pol='avg', target=0.0, weight=1.0):
        super().__init__(wvl, theta=theta, pol=pol, target=target, weight=weight)
        self.boundary = int(boundary)

    def _quantity(self, fwd):
        return fwd.Esq_value[self.boundary]

    def _seed_kw(self, fwd, dq):
        dEsq = np.zeros(fwd.Esq_value.shape, dtype=config.precision)
        dEsq[self.boundary] = dq
        return {'dEsq': dEsq}


class PeakFieldAtInterfaces(_Term):
    """Target the peak standing-wave intensity over a set of boundaries.

    The quantity is max_k |E_k|^2 over the selected boundaries -- the field at the
    hottest interface, which governs laser-induced damage.  Minimizing it (target
    0) flattens the standing wave at the interfaces.  The gradient is the
    subgradient at the per-sample argmax boundary, which equals the gradient away
    from ties.

    Parameters
    ----------
    boundaries : sequence of int, optional
        boundary indices to consider (0 is the ambient face, N the substrate
        face).  Default: every boundary.

    """

    quantity = 'Esq'

    def __init__(self, wvl, theta=0.0, pol='avg', boundaries=None,
                 target=0.0, weight=1.0):
        super().__init__(wvl, theta=theta, pol=pol, target=target, weight=weight)
        self.boundaries = None if boundaries is None else list(boundaries)

    def _selected(self, fwd):
        Esq = fwd.Esq_value
        if self.boundaries is None:
            return Esq, np.arange(Esq.shape[0])
        bidx = np.asarray(self.boundaries)
        return Esq[bidx], bidx

    def _quantity(self, fwd):
        Esq, _ = self._selected(fwd)
        return np.max(Esq, axis=0)

    def _seed_kw(self, fwd, dq):
        Esq, bidx = self._selected(fwd)
        ndc = Esq.ndim - 1
        am = np.argmax(Esq, axis=0)
        ar = np.arange(Esq.shape[0]).reshape((Esq.shape[0],) + (1,) * ndc)
        onehot = (ar == am[None]).astype(config.precision)
        dEsq_sel = onehot * dq[None]
        full = np.zeros(fwd.Esq_value.shape, dtype=config.precision)
        full[bidx] = dEsq_sel
        return {'dEsq': full}


class FieldInLayer(_Term):
    """Target the standing-wave intensity within a designated layer.

    The quantity is the mean of |E|^2 at the layer's two bounding boundaries -- a
    smooth proxy for the field inside the layer.  Drive it up (large target) for
    an active / gain layer, down (target 0) to protect a sensitive layer.

    Parameters
    ----------
    layer : int
        layer index (ambient side first); the field is sampled at boundaries
        layer and layer + 1.

    """

    quantity = 'Esq'

    def __init__(self, layer, wvl, theta=0.0, pol='avg', target=0.0, weight=1.0):
        super().__init__(wvl, theta=theta, pol=pol, target=target, weight=weight)
        self.layer = int(layer)

    def _quantity(self, fwd):
        Esq = fwd.Esq_value
        return 0.5 * (Esq[self.layer] + Esq[self.layer + 1])

    def _seed_kw(self, fwd, dq):
        dEsq = np.zeros(fwd.Esq_value.shape, dtype=config.precision)
        dEsq[self.layer] = dEsq[self.layer] + 0.5 * dq
        dEsq[self.layer + 1] = dEsq[self.layer + 1] + 0.5 * dq
        return {'dEsq': dEsq}


class MeritFunction:
    """A weighted collection of merit terms.

    The value, residual vector, and thickness gradient are the concatenation /
    sum across the terms, so a single MeritFunction can mix reflectance,
    transmittance, absorptance, and field objectives at once (the multi-objective
    design path).
    """

    __slots__ = ('terms',)

    def __init__(self, terms):
        """Collect one or more merit terms into a single objective.

        Parameters
        ----------
        terms : _Term or sequence of _Term
            the terms to sum; a lone term is accepted and wrapped in a list.

        """
        if isinstance(terms, _Term):
            terms = [terms]
        self.terms = list(terms)

    def value(self, stack):
        """Total weighted sum-of-squares merit (a scalar)."""
        return float(sum(t.value(stack) for t in self.terms))

    def residuals(self, stack):
        """Concatenated weighted residual vector across all terms."""
        if not self.terms:
            return np.zeros(0, dtype=config.precision)
        return np.concatenate([t.residuals(stack) for t in self.terms])

    def value_and_grad(self, stack, grad_fn=thickness_gradient):
        """Total merit and gradient via grad_fn (thickness_gradient default).

        Pass diff.index_gradient as grad_fn for index-variable (rugate) design.
        """
        val = 0.0
        grad = np.zeros(len(stack), dtype=config.precision)
        for t in self.terms:
            v, g = t.value_and_grad(stack, grad_fn=grad_fn)
            val = val + v
            grad = grad + g
        return float(val), grad


def as_merit(obj):
    """Normalize a term, list of terms, or MeritFunction into a MeritFunction."""
    if isinstance(obj, MeritFunction):
        return obj
    if isinstance(obj, _Term):
        return MeritFunction([obj])
    return MeritFunction(list(obj))


__all__ = [
    'Reflectance',
    'Transmittance',
    'LayerAbsorptance',
    'FieldIntensityAtBoundary',
    'PeakFieldAtInterfaces',
    'FieldInLayer',
    'MeritFunction',
    'as_merit',
]
