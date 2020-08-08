"""Tools for working with Q (Forbes) polynomials."""
from functools import lru_cache

from .conf import config
from .pupil import Pupil
from .mathops import engine as e, kronecker, gamma
from .coordinates import gridcache
from .jacobi import jacobi

MAX_ELEMENTS_IN_CACHE = 1024  # surely no one wants > 1000 terms...


def qbfs_recurrence_P(n, x, Pnm1=None, Pnm2=None, recursion_coef=None):
    """P(m+1) from oe-18-19-19700 eq. (2.6).

    Parameters
    ----------
    n : `int`
        polynomial order
    x : `numpy.ndarray`
        x values, notionally in / orthogonal over [0, 1], to evaluate at
    Pnm1 : `numpy.ndarray`, optional
        the value of this function for argument n - 1
    Pnm2 : `numpy.ndarray`, optional
        the value of this function for argument n - 2
    recursion_coef : `numpy.ndarray`, optional
        the coefficient to apply, if recursion_coef = C: evaluates C * Pnm1 - Pnm2

    Returns
    -------
    `numpy.ndarray`
        the value of the auxiliary P polynomial for given order n and point(s) x

    """
    if n == 0:
        return 2
    elif n == 1:
        return 6 - 8 * x
    else:
        if Pnm2 is None:
            Pnm2 = qbfs_recurrence_P(n - 2, x)
        if Pnm1 is None:
            Pnm1 = qbfs_recurrence_P(n - 1, x, Pnm1=Pnm2)

        if recursion_coef is None:
            recursion_coef = 2 - 4 * x

        return recursion_coef * Pnm1 - Pnm2


def qbfs_recurrence_Q(n, x, Pn=None, Pnm1=None, Pnm2=None, Qnm1=None, Qnm2=None, recursion_coef=None):
    """Q(m+1) from oe-18-19-19700 eq. (2.7).

    Parameters
    ----------
    n : `int`
        polynomial order
    x : `numpy.ndarray`
        x values, notionally in / orthogonal over [0, 1], to evaluate at
    Pnm1 : `numpy.ndarray`, optional
        the value of qbfs_recurrence_P for argument n - 1
    Pnm2 : `numpy.ndarray`, optional
        the value of qbfs_recurrence_P for argument n - 2
    Qnm1 : `numpy.ndarray`, optional
        the value of this function for argument n - 1
    Qnm2 : `numpy.ndarray`, optional
        the value of this function for argument n - 2
    recursion_coef : `numpy.ndarray`, optional
        the coefficient to apply, if recursion_coef = C: evaluates C * Pnm1 - Pnm2

    Returns
    -------
    `numpy.ndarray`
        the value of the the Qbfs polynomial for given order n and point(s) x

    """
    if n == 0:
        return e.ones_like(x)
    elif n == 1:
        return 1 / e.sqrt(19) * (13 - 16 * x)
    else:
        # allow passing of cached results
        if Pnm2 is None:
            Pnm2 = qbfs_recurrence_P(n - 2, x, recursion_coef=recursion_coef)
        if Pnm1 is None:
            Pnm1 = qbfs_recurrence_P(n - 1, x, Pnm1=Pnm2, recursion_coef=recursion_coef)
        if Pn is None:
            Pn = qbfs_recurrence_P(n, x, Pnm1=Pnm1, Pnm2=Pnm2, recursion_coef=recursion_coef)
        if Qnm2 is None:
            Qnm2 = qbfs_recurrence_Q(n - 2, x, Pn=Pn, Pnm1=Pnm1, Pnm2=Pnm2, recursion_coef=recursion_coef)
        if Qnm1 is None:
            Qnm1 = qbfs_recurrence_Q(n - 1, x, Pn=Pn, Pnm1=Pnm1, Pnm2=Pnm2, Qnm2=Qnm2, recursion_coef=recursion_coef)

        # now calculate the three-term recursion
        term1 = Pn
        term2 = g_qbfs(n - 1) * Qnm1
        term3 = h_qbfs(n - 2) * Qnm2
        numerator = term1 - term2 - term3
        denominator = f_qbfs(n)
        return numerator / denominator


@lru_cache(MAX_ELEMENTS_IN_CACHE)
def g_qbfs(n_minus_1):
    """g(m-1) from oe-18-19-19700 eq. (A.15)."""
    if n_minus_1 == 0:
        return - 1 / 2
    else:
        n_minus_2 = n_minus_1 - 1
        return - (1 + g_qbfs(n_minus_2) * h_qbfs(n_minus_2)) / f_qbfs(n_minus_1)


@lru_cache(MAX_ELEMENTS_IN_CACHE)
def h_qbfs(n_minus_2):
    """h(m-2) from oe-18-19-19700 eq. (A.14)."""
    n = n_minus_2 + 2
    return -n * (n - 1) / (2 * f_qbfs(n_minus_2))


@lru_cache(MAX_ELEMENTS_IN_CACHE)
def f_qbfs(n):
    """f(m) from oe-18-19-19700 eq. (A.16)."""
    if n == 0:
        return 2
    elif n == 1:
        return e.sqrt(19) / 2
    else:
        term1 = n * (n + 1) + 3
        term2 = g_qbfs(n - 1) ** 2
        term3 = h_qbfs(n - 2) ** 2
        return e.sqrt(term1 - term2 - term3)


class QBFSCache(object):
    """Cache of Qbfs terms evaluated over the unit circle."""
    def __init__(self, gridcache=gridcache):
        """Create a new QBFSCache instance."""
        self.Qs = {}
        self.Ps = {}
        self.gridcache = gridcache

    def get_QBFS(self, m, samples, rho_max=1):
        """Get an array of phase values for a given index, and number of samples."""
        key = self.make_key(m=m, samples=samples, rho_max=rho_max)
        try:
            Qm = self.Qs[key]
        except KeyError:
            rho = self.get_grid(samples, rho_max=rho_max)
            Pm = self.get_PBFS(m=m, samples=samples, rho_max=rho_max)
            if m > 2:
                Pnm2 = self.get_PBFS(m=m - 2, samples=samples, rho_max=rho_max)
                Pnm1 = self.get_PBFS(m=m - 1, samples=samples, rho_max=rho_max)
                Qnm2 = self.get_QBFS(m=m - 2, samples=samples, rho_max=rho_max)
                Qnm1 = self.get_QBFS(m=m - 1, samples=samples, rho_max=rho_max)
            else:
                Pnm1, Pnm2, Qnm1, Qnm2 = None, None, None, None

            coef = self.get_PBFS_recursion_coef(samples=samples, rho_max=rho_max)
            Qm = qbfs_recurrence_Q(m, rho, Pn=Pm, Pnm1=Pnm1, Pnm2=Pnm2,
                                   Qnm1=Qnm1, Qnm2=Qnm2, recursion_coef=coef)
            self.Qs[key] = Qm

        return Qm

    def get_PBFS(self, m, samples, rho_max=1):
        """Get an array of P values for a given index."""
        key = self.make_key(m=m, samples=samples, rho_max=rho_max)
        try:
            Pm = self.Ps[key]

        except KeyError:
            rho = self.get_grid(samples=samples, rho_max=rho_max)
            if m > 2:
                Pnm2 = self.get_PBFS(m - 2, samples=samples, rho_max=rho_max)
                Pnm1 = self.get_PBFS(m - 1, samples=samples, rho_max=rho_max)
            else:
                Pnm1, Pnm2 = None, None

            coef = self.get_PBFS_recursion_coef(samples=samples, rho_max=rho_max)
            Pm = qbfs_recurrence_P(m, rho, Pnm1=Pnm1, Pnm2=Pnm2, recursion_coef=coef)
            self.Ps[key] = Pm

        return Pm

    def get_PBFS_recursion_coef(self, samples, rho_max=1):
        """Get a P polynomial recursion coefficient.

        Parameters
        ----------
        samples : `int`
            number of samples
        rho_max : `float`
            max value of rho ("x" or "r") for the polynomial evaluation

        Returns
        -------
        `float`
            recursion coefficient

        """
        key = ('recursion', samples, rho_max)
        try:
            coef = self.Ps[key]
        except KeyError:
            rho = self.get_grid(samples=samples, rho_max=rho_max)
            coef = 2 - 4 * rho
            self.Ps[key] = coef

        return coef

    def get_grid(self, samples, rho_max=1):
        """Get a P polynomial recursion coefficient.

        Parameters
        ----------
        samples : `int`
            number of samples
        rho_max : `float`
            max value of rho ("x" or "r") for the polynomial evaluation

        Returns
        -------
        `numpy.ndarray`
            2D grid of radial coordinates

        """
        return self.gridcache(samples=samples, radius=rho_max, r='r -> r^2')['r']

    def __call__(self, m, samples, rho_max=1):
        """Get a P polynomial recursion coefficient.

        Parameters
        ----------
        samples : `int`
            number of samples
        rho_max : `float`
            max value of rho ("x" or "r") for the polynomial evaluation

        Returns
        -------
        `numpy.ndarray`
            Qbfs polynomial evaluated over a grid of shape (samples, samples)

        """
        return self.get_QBFS(m=m, samples=samples, rho_max=rho_max)

    def make_key(self, m, samples, rho_max):
        """Generate a key into the cache dictionaries."""
        return (m, samples, rho_max)

    def clear(self, *args):
        """Empty the cache."""
        self.Qs = {}
        self.Ps = {}
        self.grids = {}

    @property
    def nbytes(self):
        """Bytes of memory occupied by the cache."""
        n = 0
        stores = (self.Qs, self.Ps)
        for store in stores:
            for key in store:
                n += store[key].nbytes

        return n


QBFScache = QBFSCache()
config.chbackend_observers.append(QBFScache.clear)


# Qcon is defined as:
# r^4 * P_m(0,4)(2x-1)
# with x = r^2


def qcon_recurrence(n, x, Pnm1=None, Pnm2=None):
    """Recursive Qcon polynomial evaluation.

    Parameters
    ----------
    n : `int`
        polynomial order
    x : `numpy.ndarray`
        "x" coordinates, x = r^2
    Pnm1 : `numpy.ndarray`
        value of this function for argument (n-1)
    Pnm2 : `numpy.ndarray`
        value of this function for argument (n-2)

    Returns
    -------
    `numpy.ndarray`
        Value of the Qcon polynomials of order n over x

    """
    return jacobi(n, x=x, alpha=0, beta=4, Pnm1=Pnm1, Pnm2=Pnm2)


class QCONCache(object):
    """Cache of Qcon terms evaluated over the unit circle."""
    def __init__(self, gridcache=gridcache):
        """Create a new QCONCache instance."""
        self.Qs = {}
        self.Ps = {}
        self.gridcache = gridcache

    def get_QCON(self, m, samples, rho_max=1):
        """Get an array of phase values for a given index, and number of samples."""
        # TODO: update
        key = self.make_key(m=m, samples=samples, rho_max=rho_max)
        try:
            Qm = self.Qs[key]
        except KeyError:
            rho = self.get_grid(samples, rho_max=rho_max)
            if m > 2:
                Pnm2 = self.get_PJAC(m=m - 2, samples=samples, rho_max=rho_max)
                Pnm1 = self.get_PJAC(m=m - 1, samples=samples, rho_max=rho_max)
            else:
                Pnm1, Pnm2 = None, None

            Qm = qcon_recurrence(m, rho, Pnm1=Pnm1, Pnm2=Pnm2)
            self.Qs[key] = Qm

        return Qm

    def get_PJAC(self, m, samples, rho_max=1):
        """Get an array of P_n^(0,4) values for a given index."""
        # TODO: update
        key = self.make_key(m=m, samples=samples, rho_max=rho_max)
        try:
            Pm = self.Ps[key]

        except KeyError:
            rho = self.get_grid(samples, rho_max=rho_max)
            if m > 2:
                Pnm2 = self.get_PJAC(m - 2, samples=samples, rho_max=rho_max)
                Pnm1 = self.get_PJAC(m - 1, samples=samples, rho_max=rho_max)
            else:
                Pnm1, Pnm2 = None, None

            Pm = jacobi(n=m, x=rho, alpha=0, beta=4, Pnm1=Pnm1, Pnm2=Pnm2)
            self.Ps[key] = Pm

        return Pm

    def get_grid(self, samples, rho_max=1):
        """Get a grid of rho coordinates for a given number of samples."""
        return self.gridcache(samples=samples, radius=rho_max, r='r -> 2r^2 - 1')['r']

    def __call__(self, m, samples, rho_max=1):
        """Get an array of sag values for a given index, norm, and number of samples."""
        return self.get_QCON(m=m, samples=samples, rho_max=rho_max)

    def make_key(self, m, samples, rho_max):
        """Generate a key into the cache dictionaries."""
        return (m, samples, rho_max)

    def clear(self, *args):
        """Empty the cache."""
        self.Qs = {}
        self.Ps = {}
        self.grids = {}

    @property
    def nbytes(self):
        """Bytes of memory occupied by the cache."""
        n = 0
        for key in self.Qs:
            n += self.Qs[key].nbytes
            n += self.Ps[key].nbytes

        return n


QCONcache = QCONCache()
config.chbackend_observers.append(QCONcache.clear)


# Note that this class doesn't implement _name and other RichData requirements
class QPolySag1D(Pupil):
    """Base class with 1D Q polynomial logic."""

    def __init__(self, *args, **kwargs):
        """Initialize a new QBFS instance."""
        self.coefs = {}
        pass_args = {}
        if kwargs is not None:
            for key, value in kwargs.items():
                if key[0].lower() == 'a':
                    idx = int(key[1:])  # strip 'A' from index
                    self.coefs[idx] = value
                else:
                    pass_args[key] = value

        super().__init__(**pass_args)

    def build(self):
        """Use the aspheric coefficients stored in this class instance to build a sag model.

        Returns
        -------
        self : `QPolySag1D`
            this QPolySag1D instance`

        """
        self.phase = e.zeros([self.samples, self.samples], dtype=config.precision)
        ordered_terms = sorted(self.coefs)
        ordered_values = [self.coefs[v] for v in ordered_terms]
        for term, coef in zip(ordered_terms, ordered_values):
            if coef == 0:
                continue
            else:
                self.phase += coef * self._cache(term, self.samples)


class QBFSSag(QPolySag1D):
    """Qbfs polynomials evaluated over a grid."""
    _name = 'Qbfs'
    _cache = QBFScache
    """Qbfs aspheric surface sag, excluding base sphere."""

    def build(self):
        """Use the aspheric coefficients stored in this class instance to build a sag model.

        Returns
        -------
        self : `QBFSSag`
            this QBFSSag instance`

        """
        super().build()
        coef = self._cache.gridcache(samples=self.samples, radius=1, r='r -> r^2 (1-r^2)')['r']
        self.phase *= coef


class QCONSag(QPolySag1D):
    """Qcon polynomials evaluated over a grid."""
    _name = 'Qcon'
    _cache = QCONcache
    """Qcon aspheric surface sag, excluding base sphere."""

    def build(self):
        """Use the aspheric coefficients stored in this class instance to build a sag model.

        Returns
        -------
        self : `QCONSag`
            this QCONSag instance`

        """
        super().build()
        coef = self._cache.gridcache(samples=self.samples, radius=1, r='r -> r^4')['r']
        self.phase *= coef


def abc_q2d(n, m):
    """A, B, C terms for 2D-Q polynomials.  oe-20-3-2483 Eq. (A.3).

    Parameters
    ----------
    n : `int`
        radial order
    m : `int`
        azimuthal order

    Returns
    -------
    `float`, `float`, `float`
        A, B, C

    """
    # D is used everywhere
    D = (4 * n ** 2 - 1) * (m + n - 2) * (m + 2 * n - 3)

    # A
    term1 = (2 * n - 1) * (m + 2 * n - 2)
    term2 = (4 * n * (m + n - 2) + (m - 3) * (2 * m - 1))
    A = (term1 * term2) / D

    # B
    num = -2 * (2 * n - 1) * (m + 2 * n - 3) * (m + 2 * n - 2) * (m + 2 * n - 1)
    B = num / D

    # C
    num = n * (2 * n - 3) * (m + 2 * n - 1) * (2 * m + 2 * n - 3)
    C = num / D

    return A, B, C


def G_q2d(n, m):
    """G term for 2D-Q polynomials.  oe-20-3-2483 Eq. (A.15).

    Parameters
    ----------
    n : `int`
        radial order
    m : `int`
        azimuthal order

    Returns
    -------
    `float`
        G

    """
    if n == 0:
        num = e.scipy.special.factorial2(2 * m - 1)
        den = 2 ** (m + 1) * e.scipy.special.factorial(m - 1)
        return num / den
    elif n > 0 and m == 1:
        t1num = (2 * n ** 2 - 1) * (n ** 2 - 1)
        t1den = 8 * (4 * n ** 2 - 1)
        term1 = -t1num / t1den
        term2 = 1 / 24 * kronecker(n, 1)
        return term1 + term2  # this is minus in the paper
    else:
        # nt1 = numerator term 1, d = denominator...
        nt1 = 2 * n * (m + n - 1) - m
        nt2 = (n + 1) * (2 * m + 2 * n - 1)
        num = nt1 * nt2
        dt1 = (m + 2 * n - 2) * (m + 2 * n - 1)
        dt2 = (m + 2 * n) * (2 * n + 1)
        den = dt1 * dt2

        term1 = num / den  # there is a leading negative in the paper
        return term1 * gamma(n, m)


def F_q2d(n, m):
    """F term for 2D-Q polynomials.  oe-20-3-2483 Eq. (A.13).

    Parameters
    ----------
    n : `int`
        radial order
    m : `int`
        azimuthal order

    Returns
    -------
    `float`
        F

    """
    if n == 0:
        num = m ** 2 * e.scipy.special.factorial2(2 * m - 3)
        den = 2 ** (m + 1) * e.scipy.special.factorial(m - 1)
        return num / den
    elif n > 0 and m == 1:
        t1num = 4 * (n - 1) ** 2 * n ** 2 + 1
        t1den = 8 * (2 * n - 1) ** 2
        term1 = t1num / t1den
        term2 = 11 / 32 * kronecker(n, 1)
        return term1 + term2
    else:
        Chi = m + n - 2
        nt1 = 2 * n * Chi * (3 - 5 * m + 4 * n * Chi)
        nt2 = m ** 2 * (3 - m + 4 * n * Chi)
        num = nt1 + nt2

        dt1 = (m + 2 * n - 3) * (m + 2 * n - 2)
        dt2 = (m + 2 * n - 1) * (2 * n - 1)
        den = dt1 * dt2

        term1 = num / den
        return term1 * gamma(n, m)


def g_q2d(nm1, m):
    """Lowercase g term for 2D-Q polynomials.  oe-20-3-2483 Eq. (A.18a).

    Parameters
    ----------
    nm1 : `int`
        radial order less one (n - 1)
    m : `int`
        azimuthal order

    Returns
    -------
    `float`
        g

    """
    return G_q2d(nm1, m) / f_q2d(nm1, m)


def f_q2d(n, m):
    """Lowercase f term for 2D-Q polynomials.  oe-20-3-2483 Eq. (A.18b).

    Parameters
    ----------
    nm1 : `int`
        radial order
    m : `int`
        azimuthal order

    Returns
    -------
    `float`
        f

    """
    if n == 0:
        return e.sqrt(F_q2d(n=0, m=m))
    else:
        return e.sqrt(F_q2d(n, m) - g_q2d(n-1, m) ** 2)


def q2d_recurrence_P(n, m, x, Pnm1=None, Pnm2=None):
    """Auxiliary polynomial P to the 2DQ polynomials (Q).  oe-20-3-2483 Eq. (A.17).

    Parameters
    ----------
    n : `int`
        radial order
    m : `int`
        azimuthal order
    x : `numpy.ndarray`
        spatial coordinates, x = r^4  # TODO: (docs) check this transformation
    Pnm1 : `numpy.ndarray`
        value of this function for argument n - 1
    Pnm2 : `numpy.ndarray`
        value of this function for argument n - 2

    Returns
    -------
    `numpy.ndarray`
        P polynomial evaluated over x

    """
    if m == 0:
        return qbfs_recurrence_P(n=n, x=x, Pnm1=Pnm1, Pnm2=Pnm2)
    elif n == 0:
        return 1 / 2
    elif n == 1:
        if m == 1:
            return 1 - x / 2
        elif m < 1:
            raise ValueError('2D-Q auxiliary polynomial is undefined for n=1, m < 1')
        else:
            return m - (1 / 2) - (m - 1) * x
    elif m == 1 and (n == 2 or n == 3):
        if n == 2:
            num = 3 - x * (12 - 8 * x)
            den = 6
            return num / den
        if n == 3:
            numt1 = 5 - x
            numt2 = 60 - x * (120 - 64 * x)
            num = numt1 * numt2
            den = 10
            return num / den
    else:
        if Pnm2 is None:
            Pnm2 = q2d_recurrence_P(n=n-2, m=m, x=x)
        if Pnm1 is None:
            Pnm1 = q2d_recurrence_P(n=n-1, m=m, x=x, Pnm1=Pnm2)

        Anm, Bnm, Cnm = abc_q2d(n, m)
        term1 = Anm + Bnm * x
        term2 = Pnm1
        term3 = Cnm * Pnm2
        return term1 * term2 - term3


def q2d_recurrence_Q(n, m, x, Pn=None, Qnm1=None, Pnm1=None, Pnm2=None):
    """2DQ polynomials (Q).  oe-20-3-2483 Eq. (A.22).

    Parameters
    ----------
    n : `int`
        radial order
    m : `int`
        azimuthal order
    x : `numpy.ndarray`
        spatial coordinates, x = r^4  # TODO: (docs) check this transformation
    Pn : `numpy.ndarray`
        value of this function for same order n
    Qnm1 : `numpy.ndarray`
        value of this function for argument n - 1
    Pnm1 : `numpy.ndarray`
        value of the paired P function for n - 1
    Pnm2 : `numpy.ndarray`
        value of the paired P function for n - 2

    Returns
    -------
    `numpy.ndarray`
        P polynomial evaluated over x

    """
    if n == 0:
        return 1 / (2 * f_q2d(0, m))
    elif m == 0:
        return qbfs_recurrence_Q(n=n, x=x, Pn=Pn, Pnm1=Pnm1, Pnm2=Pnm2, Qnm1=Qnm1)

    if Pnm2 is None:
        Pnm2 = q2d_recurrence_P(n=n-2, m=m, x=x)
    if Pnm1 is None:
        Pnm1 = q2d_recurrence_P(n=n-1, m=m, x=x, Pnm1=Pnm2)
    if Pn is None:
        if n == 0:
            Pnm = f_q2d(0, m) * q2d_recurrence_Q(n=0, m=m, x=x)
        else:
            Pnm = q2d_recurrence_P(n=n, m=m, x=x, Pnm1=Pnm1, Pnm2=Pnm2)

    if Qnm1 is None:
        Qnm1 = q2d_recurrence_Q(n=n-1, m=m, x=x, Pnm=Pnm1, Pnm1=Pnm2)

    return (Pnm - g_q2d(n-1, m) * Qnm1) / f_q2d(n, m)
