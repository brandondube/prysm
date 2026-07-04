"""Analytic adjoints for coating transfer matrices.

Complex cotangents use dF = Re(conj(c_z) dz).  Gradients are summed over the
broadcast wavelength / angle sample axes.
"""

from prysm.conf import config
from prysm.mathops import np
from prysm.thinfilm import _cos_snell

from .stack import (
    _resolve,
    _admittance,
    _char_matrix,
    _matvec,
    stack_characteristic_matrices,
    forward_products,
    backward_products,
)


def _dchar_dbeta(beta, eta):
    """Derivative of the characteristic matrix w.r.t. its phase thickness beta."""
    cosb = np.cos(beta) + 0j
    sinb = np.sin(beta)
    row0 = np.stack([-sinb, -1j * cosb / eta], axis=-1)
    row1 = np.stack([-1j * eta * cosb, -sinb], axis=-1)
    return np.stack([row0, row1], axis=-2)


def _dchar_deta(beta, eta):
    """Derivative of the characteristic matrix w.r.t. its admittance eta."""
    sinb = np.sin(beta)
    zero = np.zeros_like(sinb + 0j)
    row0 = np.stack([zero, 1j * sinb / (eta * eta)], axis=-1)
    row1 = np.stack([-1j * sinb, zero], axis=-1)
    return np.stack([row0, row1], axis=-2)


def char_matrix_vjp(beta, eta, M_bar):
    """Pull a matrix cotangent back to beta and eta cotangents.

    Parameters
    ----------
    beta, eta : ndarray
        phase thickness and admittance, shape (calc,).
    M_bar : ndarray
        cotangent of the characteristic matrix, shape (calc, 2, 2).

    Returns
    -------
    (ndarray, ndarray)
        c_beta and c_eta.

    """
    dMdb = _dchar_dbeta(beta, eta)
    dMde = _dchar_deta(beta, eta)
    c_beta = np.sum(np.conj(dMdb) * M_bar, axis=(-2, -1))
    c_eta = np.sum(np.conj(dMde) * M_bar, axis=(-2, -1))
    return c_beta, c_eta


class ForwardEval:
    """Cached forward evaluation of a stack."""

    __slots__ = ('stack', 'wvl', 'theta0', 'pol', 'matrices', 'L', 'R', 'M',
                 'eta0', 'eta_sub', 'r', 't', 'v_sub', 'E', 'H',
                 'betas', 'etas', 'dbeta_dd', 'n0', 'ns', 'costs')

    def __init__(self, stack, wvl, theta0, pol):
        """Evaluate and cache stack state.

        Parameters
        ----------
        stack : Stack
        wvl : float or ndarray
            wavelength(s), microns.
        theta0 : float or ndarray
            angle(s) of incidence in the ambient medium, radians.
        pol : str
            polarization, 'p' or 's'.

        """
        pol = pol.lower()
        if pol not in ('p', 's'):
            raise ValueError("unknown polarization, use 'p' or 's'")
        self.stack = stack
        self.wvl = wvl
        self.theta0 = theta0
        self.pol = pol

        n0 = _resolve(stack.ambient_index, wvl)
        nsub = _resolve(stack.substrate_index, wvl)
        cost0 = np.cos(theta0)
        cost_sub = _cos_snell(n0, nsub, theta0)
        self.n0 = n0
        self.eta0 = _admittance(n0, cost0, pol)
        self.eta_sub = _admittance(nsub, cost_sub, pol)

        # per-layer phase/admittance and the d/d(thickness) of beta
        betas = []
        etas = []
        dbeta_dd = []
        ns = []
        costs = []
        for n, d in zip(stack.resolved_indices(wvl), stack.thicknesses):
            cost = _cos_snell(n0, n, theta0)
            betas.append((2 * np.pi * n * d * cost) / wvl)
            etas.append(_admittance(n, cost, pol))
            dbeta_dd.append((2 * np.pi * n * cost) / wvl)
            ns.append(n)
            costs.append(cost)
        self.betas = betas
        self.etas = etas
        self.dbeta_dd = dbeta_dd
        self.ns = ns
        self.costs = costs

        self.matrices = stack_characteristic_matrices(stack, wvl, theta0, pol)
        self.L = forward_products(self.matrices)
        self.R = backward_products(self.matrices)
        self.M = self.R[0]

        B = self.M[..., 0, 0] + self.M[..., 0, 1] * self.eta_sub
        C = self.M[..., 1, 0] + self.M[..., 1, 1] * self.eta_sub
        denom = self.eta0 * B + C
        self.r = (self.eta0 * B - C) / denom
        self.t = 2 * self.eta0 / denom

        self.v_sub = np.stack([self.t, self.t * self.eta_sub], axis=-1)
        fields = np.stack([_matvec(Rk, self.v_sub) for Rk in self.R], axis=0)
        self.E = fields[..., 0]
        self.H = fields[..., 1]

    @property
    def R_value(self):
        """Reflectance abs(r)^2."""
        return np.abs(self.r) ** 2

    @property
    def T_value(self):
        """Transmittance with the tilted-admittance flux factor."""
        return np.real(self.eta_sub) / np.real(self.eta0) * np.abs(self.t) ** 2

    @property
    def A_value(self):
        """Per-layer absorptance, shape (N, calc)."""
        flux = np.real(self.E * np.conj(self.H)) / np.real(self.eta0)
        return flux[:-1] - flux[1:]

    @property
    def Esq_value(self):
        """Standing-wave intensity abs(E)^2 at each boundary, shape (N+1, calc)."""
        return np.abs(self.E) ** 2


def forward_eval(stack, wvl, theta0, pol):
    """Build a ForwardEval for one sample set."""
    return ForwardEval(stack, wvl, theta0, pol)


def _rt_assembly_cotangent(fwd, r_bar, t_bar):
    """Assembled-matrix cotangent from amplitude cotangents."""
    eta0 = fwd.eta0
    eta_sub = fwd.eta_sub
    M = fwd.M
    B = M[..., 0, 0] + M[..., 0, 1] * eta_sub
    C = M[..., 1, 0] + M[..., 1, 1] * eta_sub
    den = eta0 * B + C
    r = fwd.r
    t = fwd.t

    # holomorphic partials of r, t w.r.t. B, C
    dr_dB = eta0 * (1 - r) / den
    dr_dC = -(1 + r) / den
    dt_dB = -t * eta0 / den
    dt_dC = -t / den

    B_bar = np.conj(dr_dB) * r_bar + np.conj(dt_dB) * t_bar
    C_bar = np.conj(dr_dC) * r_bar + np.conj(dt_dC) * t_bar

    M_bar = np.zeros_like(M)
    ce = np.conj(eta_sub)
    M_bar[..., 0, 0] = B_bar
    M_bar[..., 0, 1] = ce * B_bar
    M_bar[..., 1, 0] = C_bar
    M_bar[..., 1, 1] = ce * C_bar
    return M_bar


def _physical_to_amplitude_cotangents(fwd, dR, dT):
    """Map real reflectance/transmittance cotangents to r_bar, t_bar."""
    shape = fwd.r.shape
    r_bar = np.zeros(shape, dtype=fwd.r.dtype)
    t_bar = np.zeros(shape, dtype=fwd.t.dtype)
    if dR is not None:
        r_bar = r_bar + 2 * dR * fwd.r
    if dT is not None:
        factor = np.real(fwd.eta_sub) / np.real(fwd.eta0)
        t_bar = t_bar + 2 * factor * dT * fwd.t
    return r_bar, t_bar


def _field_cotangents(fwd, dA, dEsq):
    """Field-quantity cotangents to E and H cotangents."""
    nb = len(fwd.R)              # N + 1 boundaries
    calc_shape = fwd.E.shape[1:]
    E_bar = np.zeros((nb,) + calc_shape, dtype=fwd.E.dtype)
    H_bar = np.zeros((nb,) + calc_shape, dtype=fwd.H.dtype)

    if dA is not None:
        # A_k = flux_k - flux_{k+1};  flux_k = Re(E_k conj(H_k))/Re(eta0).
        # boundary k's flux appears with +1 in A_k and -1 in A_{k-1}.
        re0 = np.real(fwd.eta0)
        flux_bar = np.zeros((nb,) + calc_shape, dtype=config.precision)
        flux_bar[:-1] = flux_bar[:-1] + dA           # +1 from A_k
        flux_bar[1:] = flux_bar[1:] - dA             # -1 from A_{k-1}
        E_bar = E_bar + (flux_bar / re0) * fwd.H
        H_bar = H_bar + (flux_bar / re0) * fwd.E

    if dEsq is not None:
        E_bar = E_bar + 2 * dEsq * fwd.E

    return E_bar, H_bar


def assembly_cotangent(fwd, dR=None, dT=None):
    """Assembled-matrix cotangent for an R/T merit."""
    r_bar, t_bar = _physical_to_amplitude_cotangents(fwd, dR, dT)
    return _rt_assembly_cotangent(fwd, r_bar, t_bar)


def layer_cotangents(fwd, dR=None, dT=None, dA=None, dEsq=None):
    """Per-layer beta and eta cotangents for a scalar merit.

    Returns
    -------
    (list, list)
        c_beta and c_eta, each a length-N list of (calc,) arrays.

    """
    N = len(fwd.matrices)
    matrices = fwd.matrices
    R = fwd.R

    # amplitude-path cotangent on the assembled matrix M = R[0]
    r_bar, t_bar = _physical_to_amplitude_cotangents(fwd, dR, dT)

    # field-path cotangents: f_k = R_k v_sub feeds v_sub (-> t) and R_k.
    E_bar, H_bar = _field_cotangents(fwd, dA, dEsq)
    have_fields = (dA is not None) or (dEsq is not None)

    # G_k holds the cotangent on the partial product R_k from boundary field f_k.
    G = [None] * N
    if have_fields:
        v = fwd.v_sub
        v_bar = np.zeros_like(v)
        cvb = np.conj(v)
        for k in range(N + 1):
            f_bar = np.stack([E_bar[k], H_bar[k]], axis=-1)
            # v_sub cotangent: c_v += R_k^H f_bar
            Rk_H = np.conj(np.swapaxes(R[k], -1, -2))
            v_bar = v_bar + _matvec(Rk_H, f_bar)
            # R_k cotangent (outer product); R[N] = I is constant, skip.
            if k < N:
                G[k] = f_bar[..., :, None] * cvb[..., None, :]
        # v_sub = t [1, eta_sub]  ->  t_bar
        t_bar = t_bar + v_bar[..., 0] + np.conj(fwd.eta_sub) * v_bar[..., 1]

    if N == 0:
        return [], []

    M_bar = _rt_assembly_cotangent(fwd, r_bar, t_bar)
    # fold the amplitude-path assembly cotangent into G_0 (since M == R[0]).
    G[0] = M_bar if G[0] is None else (G[0] + M_bar)

    # O(N) scan: W_j = G_j + M_{j-1}^H W_{j-1};  c_{M_j} = W_j R_{j+1}^H.
    c_beta = [None] * N
    c_eta = [None] * N
    W = None
    for j in range(N):
        Gj = G[j] if G[j] is not None else np.zeros_like(M_bar)
        if j == 0:
            W = Gj
        else:
            Mjm1_H = np.conj(np.swapaxes(matrices[j - 1], -1, -2))
            W = Gj + Mjm1_H @ W
        Rj1_H = np.conj(np.swapaxes(R[j + 1], -1, -2))
        c_Mj = W @ Rj1_H
        c_beta[j], c_eta[j] = char_matrix_vjp(fwd.betas[j], fwd.etas[j], c_Mj)
    return c_beta, c_eta


def thickness_gradient(fwd, dR=None, dT=None, dA=None, dEsq=None):
    """Gradient of a scalar merit w.r.t. every layer thickness.

    Returns
    -------
    ndarray
        real gradient, one value per layer.

    """
    N = len(fwd.matrices)
    c_beta, _ = layer_cotangents(fwd, dR=dR, dT=dT, dA=dA, dEsq=dEsq)
    grad = np.zeros(N, dtype=config.precision)
    for j in range(N):
        grad[j] = np.sum(np.real(np.conj(c_beta[j]) * fwd.dbeta_dd[j]))
    return grad


def index_gradient(fwd, dR=None, dT=None, dA=None, dEsq=None):
    """Gradient of a scalar merit w.r.t. every layer index.

    Returns
    -------
    ndarray
        real gradient, one value per layer.

    """
    N = len(fwd.matrices)
    c_beta, c_eta = layer_cotangents(fwd, dR=dR, dT=dT, dA=dA, dEsq=dEsq)
    wvl = fwd.wvl
    n0 = fwd.n0
    sin0_sq = np.sin(fwd.theta0) ** 2
    th = np.asarray(fwd.stack.thicknesses, dtype=config.precision)
    grad = np.zeros(N, dtype=config.precision)
    for j in range(N):
        n = fwd.ns[j]
        cost = fwd.costs[j]
        # d(cos theta_j)/d n_j from n0 sin0 = n_j sin_j (Snell invariant)
        dcost_dn = (n0 * n0 * sin0_sq) / (n ** 3 * cost)
        dbeta_dn = (2 * np.pi * th[j] / wvl) * (cost + n * dcost_dn)
        if fwd.pol == 'p':
            deta_dn = 1.0 / cost - n * dcost_dn / (cost * cost)
        else:
            deta_dn = cost + n * dcost_dn
        contrib = (np.conj(c_beta[j]) * dbeta_dn
                   + np.conj(c_eta[j]) * deta_dn)
        grad[j] = np.sum(np.real(contrib))
    return grad


__all__ = [
    'ForwardEval',
    'forward_eval',
    'char_matrix_vjp',
    'assembly_cotangent',
    'layer_cotangents',
    'thickness_gradient',
    'index_gradient',
]
