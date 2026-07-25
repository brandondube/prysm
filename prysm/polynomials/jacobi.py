"""High performance / recursive jacobi polynomial calculation."""
from prysm.mathops import np

from functools import lru_cache

from ._clenshaw import _initialize_alphas, _clenshaw_sum, _clenshaw_sum_der  # NOQA
from ._recurrence import _seq_by_recurrence, _seq_by_recurrence_with_der


def weight(alpha, beta, x):
    """The weight function of the jacobi polynomials for a given alpha, beta value."""
    return (1 - x) ** alpha * (1 + x) ** beta


@lru_cache(512)
def recurrence_abc(n, alpha, beta):
    """See A&S online - https://dlmf.nist.gov/18.9 .

    Pn = (an-1 x + bn-1) Pn-1 - cn-1 * Pn-2

    This function makes a, b, c for the given n,
    i.e. to get a(n-1), do recurrence_abc(n-1)

    """
    aplusb = alpha+beta
    if n == 0 and (aplusb == 0 or aplusb == -1):
        A = 1/2 * (alpha + beta) + 1
        B = 1/2 * (alpha - beta)
        C = 1
    else:
        Anum = (2 * n + alpha + beta + 1) * (2 * n + alpha + beta + 2)
        Aden = 2 * (n + 1) * (n + alpha + beta + 1)
        A = Anum/Aden

        Bnum = (alpha**2 - beta**2) * (2 * n + alpha + beta + 1)
        Bden = 2 * (n+1) * (n + alpha + beta + 1) * (2 * n + alpha + beta)
        B = Bnum / Bden

        Cnum = (n + alpha) * (n + beta) * (2 * n + alpha + beta + 2)
        Cden = (n + 1) * (n + alpha + beta + 1) * (2 * n + alpha + beta)
        C = Cnum / Cden

    return A, B, C


def jacobi(n, alpha, beta, x):
    """Jacobi polynomial of order n with weight parameters alpha and beta.

    Parameters
    ----------
    n : int
        polynomial order
    alpha : float
        first weight parameter
    beta : float
        second weight parameter
    x : ndarray
        x coordinates to evaluate at

    Returns
    -------
    ndarray
        jacobi polynomial evaluated at the given points

    """
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        term1 = alpha + 1
        term2 = alpha + beta + 2
        term3 = (x - 1) / 2
        return term1 + term2 * term3

    Pnm1 = alpha + 1 + (alpha + beta + 2) * ((x - 1) / 2)
    A, B, C = recurrence_abc(1, alpha, beta)
    Pn = (A * x + B) * Pnm1 - C  # no C * Pnm2 =because Pnm2 = 1
    if n == 2:
        return Pn

    for i in range(3, n+1):
        Pnm2, Pnm1 = Pnm1, Pn
        A, B, C = recurrence_abc(i-1, alpha, beta)
        Pn = (A * x + B) * Pnm1 - C * Pnm2

    return Pn


def jacobi_with_der(n, alpha, beta, x):
    """Jacobi polynomial and first derivative of order n.

    This uses the differentiated three-term recurrence directly, so callers
    that need both P_n and P_n' do not need separate Jacobi recurrences.

    Parameters
    ----------
    n : int
        polynomial order
    alpha : float
        first weight parameter
    beta : float
        second weight parameter
    x : ndarray
        x coordinates to evaluate at

    Returns
    -------
    ndarray, ndarray
        P_n and dP_n/dx evaluated at the given points

    """
    if n < 0:
        return jacobi(n, alpha, beta, x), jacobi_der(n, alpha, beta, x)

    if n == 0:
        return np.ones_like(x), np.zeros_like(x)

    dP1 = 0.5 * (alpha + beta + 2)
    P1 = alpha + 1 + (alpha + beta + 2) * ((x - 1) / 2)
    if n == 1:
        return P1, np.ones_like(x) * dP1

    Pnm2 = np.ones_like(x)
    dPnm2 = np.zeros_like(x)
    Pnm1 = P1
    dPnm1 = np.ones_like(x) * dP1
    for i in range(2, n + 1):
        A, B, C = recurrence_abc(i - 1, alpha, beta)
        lin = A * x + B
        Pn = lin * Pnm1 - C * Pnm2
        dPn = A * Pnm1 + lin * dPnm1 - C * dPnm2
        Pnm2, Pnm1 = Pnm1, Pn
        dPnm2, dPnm1 = dPnm1, dPn

    return Pn, dPn


def _jacobi_pd_step(alpha, beta, x):
    """Build the joint (value, derivative) step for jacobi_seq_with_der."""
    def step(k, Pnm1, Pnm2, Dnm1, Dnm2):
        A, B, C = recurrence_abc(k - 1, alpha, beta)
        lin = A * x + B
        Pn = lin * Pnm1 - C * Pnm2
        Dn = A * Pnm1 + lin * Dnm1 - C * Dnm2
        return Pn, Dn
    return step


def jacobi_seq(ns, alpha, beta, x):
    """Jacobi polynomials of orders ns with weight parameters alpha and beta.

    Parameters
    ----------
    ns : iterable
        sorted polynomial orders to return, e.g. [1, 3, 5, 7, ...]
    alpha : float
        first weight parameter
    beta : float
        second weight parameter
    x : ndarray
        x coordinates to evaluate at

    Returns
    -------
    ndarray
        has shape (len(ns),) followed by x.shape
        e.g., for 5 modes and x of dimension 100x100,
        return has shape (5, 100, 100)

    """
    def step(k, Pnm1, Pnm2):
        A, B, C = recurrence_abc(k - 1, alpha, beta)
        return (A * x + B) * Pnm1 - C * Pnm2

    seed1 = alpha + 1 + (alpha + beta + 2) * ((x - 1) / 2)
    return _seq_by_recurrence(ns, x, 1, seed1, step)


def jacobi_seq_with_der(ns, alpha, beta, x):
    """Jacobi polynomials and first derivatives for orders ns.

    Parameters
    ----------
    ns : iterable
        sorted polynomial orders to return
    alpha : float
        first weight parameter
    beta : float
        second weight parameter
    x : ndarray
        x coordinates to evaluate at

    Returns
    -------
    ndarray, ndarray
        P_n and dP_n/dx arrays, each shaped as (len(ns),) followed by x.shape

    """
    dP1 = 0.5 * (alpha + beta + 2)
    P1 = alpha + 1 + (alpha + beta + 2) * ((x - 1) / 2)
    step = _jacobi_pd_step(alpha, beta, x)
    return _seq_by_recurrence_with_der(ns, x, 1, P1, 0, dP1, step)


def jacobi_der(n, alpha, beta, x):
    """First derivative of Pn with respect to x, at points x.

    Parameters
    ----------
    n : int
        polynomial order
    alpha : float
        first weight parameter
    beta : float
        second weight parameter
    x : ndarray
        x coordinates to evaluate at

    Returns
    -------
    ndarray
        jacobi polynomial evaluated at the given points

    """
    # see https://dlmf.nist.gov/18.9
    # dPn = (1/2) (n + a + b + 1)P_{n-1}^{a+1,b+1}
    # first two terms are specialized for speed
    if n == 0:
        return np.zeros_like(x)
    if n == 1:
        return np.ones_like(x) * (0.5 * (n + alpha + beta + 1))

    Pn = jacobi(n-1, alpha+1, beta+1, x)
    coef = 0.5 * (n + alpha + beta + 1)
    return coef * Pn


def jacobi_der_seq(ns, alpha, beta, x):
    """First partial derivative of Pn w.r.t. x for order ns, i.e. P_n'.

    Parameters
    ----------
    ns : iterable
        sorted orders to return, e.g. [1, 2, 3, 10] returns P1', P2', P3', P10'
    alpha : float
        first weight parameter
    beta : float
        second weight parameter
    x : ndarray
        x coordinates to evaluate at

    Returns
    -------
    ndarray
        has shape (len(ns),) followed by x.shape
        e.g., for 5 modes and x of dimension 100x100,
        return has shape (5, 100, 100)

    """
    # dPn/dx = 0.5 (n+a+b+1) P_{n-1}^{a+1,b+1}; see jacobi_der.
    if not hasattr(ns, '__len__'):
        ns = list(ns)
    out = np.empty((len(ns), *x.shape), dtype=x.dtype)
    lead = 0
    while lead < len(ns) and ns[lead] == 0:
        out[lead] = 0
        lead += 1

    if lead == len(ns):
        return out

    shifted = [n - 1 for n in ns[lead:]]
    Pns = jacobi_seq(shifted, alpha + 1, beta + 1, x)
    for i, n in enumerate(ns[lead:], start=lead):
        out[i] = Pns[i - lead] * (0.5 * (n + alpha + beta + 1))

    return out


def jacobi_sum_clenshaw(s, alpha, beta, x, alphas=None):
    """Compute a weighted sum of Jacobi polynomials using Clenshaw's method.

    Parameters
    ----------
    s : iterable
        weights in ascending order, beginning with P0, then P1, etc.
        must be fully dense when iterated
    alpha : float
        first Jacobi shape parameter
    beta : float
        second Jacobi shape parameter
    x : ndarray or float_like
        coordinates to evaluate the sum at,
        orthogonal over [-1,1]
    alphas : ndarray, optional
        array to store the alpha sums in, alphas[0] contains the sum and is returned
        if not None, alphas should be of shape (len(s), x.shape)
        see _initialize_alphas if you desire more information

    Returns
    -------
    ndarray
        weighted sum of Jacobi polynomials

    """
    # A&S notation: Pn = (a x + b)Pn-1 - cPn-2 (note: Forbes swaps a and b).
    alphas = _initialize_alphas(s, x, alphas)

    def lin(n):
        a, b, _ = recurrence_abc(n, alpha, beta)
        return a * x + b

    def c_fn(n):
        return recurrence_abc(n, alpha, beta)[2]

    _clenshaw_sum(s, lin, c_fn, alphas)
    return alphas[0]


def jacobi_sum_clenshaw_der(s, alpha, beta, x, j=1, alphas=None):
    """Compute a weighted sum of partial derivatives w.r.t. x of Jacobi polynomials using Clenshaw's method.

    Notes
    -----
    If the polynomial values and their derivatives are desired, pass
    alphas instead of leaving it None.  alphas[0,0] will contain the
    sum of the polynomials, alphas[1,0] the sum of the first derivative,
    and so on.

    Parameters
    ----------
    s : iterable
        weights in ascending order, beginning with P0, then P1, etc.
        must be fully dense when iterated
    alpha : float
        first Jacobi shape parameter
    beta : float
        second Jacobi shape parameter
    x : ndarray or float_like
        coordinates to evaluate the sum at,
        orthogonal over [-1,1]
    j : int
        derivative order to compute
    alphas : ndarray, optional
        array to store the alpha sums in,
        alphas[n] is the nth order derivative alpha terms
        with n=0 being the non-derivative terms.

        for a given n, the value of alphas[0] is the nth derivative of the surface sum
        if not None, alphas should have shape (j+1, len(s)) followed by x.shape
        see _initialize_alphas if you desire more information

    Returns
    -------
    ndarray
        alphas array, see alphas parameter documentation for meaning

    """
    # alphas is dual indexed by alphas[j][n] (j=derivative, n=order)
    alphas = _initialize_alphas(s, x, None, j=j)

    def lin(n):
        a, b, _ = recurrence_abc(n, alpha, beta)
        return a * x + b

    def lin_x(n):
        # coefficient of x in the linear factor; A&S uses (a x + b), so it's a
        return recurrence_abc(n, alpha, beta)[0]

    def c_fn(n):
        return recurrence_abc(n, alpha, beta)[2]

    _clenshaw_sum_der(s, lin, lin_x, c_fn, alphas, j)
    return alphas


def jacobi_radial_sum(coefs, ns, alpha, beta, x, y, normalization_radius):
    """Evaluate a weighted radial Jacobi polynomial sum on x, y points."""
    ns = tuple(ns)
    if not ns:
        return np.zeros_like(x)
    R = float(normalization_radius)
    u = 2.0 * (x * x + y * y) / (R * R) - 1.0
    P = jacobi_seq(ns, alpha, beta, u)
    z = np.zeros_like(x)
    for i, c in enumerate(coefs):
        if c == 0.0:
            continue
        z = z + c * P[i]
    return z


def jacobi_radial_sum_der_xy(coefs, ns, alpha, beta, x, y,
                             normalization_radius):
    """Evaluate a radial Jacobi sum and its Cartesian derivatives."""
    ns = tuple(ns)
    if not ns:
        z = np.zeros_like(x)
        return z, z, np.zeros_like(y)
    R = float(normalization_radius)
    inv_Rsq = 1.0 / (R * R)
    u = 2.0 * (x * x + y * y) * inv_Rsq - 1.0
    P = jacobi_seq(ns, alpha, beta, u)
    Pp = jacobi_der_seq(ns, alpha, beta, u)
    z = np.zeros_like(x)
    dzdu = np.zeros_like(x)
    for i, c in enumerate(coefs):
        if c == 0.0:
            continue
        z = z + c * P[i]
        dzdu = dzdu + c * Pp[i]
    dzdx = dzdu * (4.0 * x * inv_Rsq)
    dzdy = dzdu * (4.0 * y * inv_Rsq)
    return z, dzdx, dzdy
