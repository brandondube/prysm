"""Tools for working with Q (Forbes) polynomials."""
from functools import lru_cache
from collections import defaultdict

from .conf import config
from .pupil import Pupil
from .mathops import engine as e
from .coordinates import make_rho_phi_grid


def qbfs_recurrence_P(n, x, Pnm1=None, Pnm2=None):
    if n == 0:
        return 2
    elif n == 1:
        return 6 - 8 * x
    else:
        if Pnm1 is None:
            Pnm1 = qbfs_recurrence_P(n - 1, x)
        if Pnm2 is None:
            Pnm2 = qbfs_recurrence_P(n - 2, x)
        return (2 - 4 * x) * Pnm1 - Pnm2


def qbfs_recurrence_Q(n, x, Pn=None, Pnm1=None, Pnm2=None, Qnm1=None, Qnm2=None):
    if n == 0:
        return e.ones_like(x, dtype=config.precision)
    elif n == 1:
        return 1 / e.sqrt(19) * (13 - 16 * x)
    else:
        if Pn is None:
            if Pnm1 is None:
                Pnm1 = qbfs_recurrence_P(n - 1, x)
            if Pnm2 is None:
                Pnm2 = qbfs_recurrence_P(n - 2, x)
            Pn = qbfs_recurrence_P(n, x, Pnm1=Pnm1, Pnm2=Pnm2)
        if Qnm1 is None:
            Qnm1 = qbfs_recurrence_Q(n - 1, x)
        if Qnm2 is None:
            Qnm2 = qbfs_recurrence_Q(n - 2, x)
        term1 = Pn
        term2 = g(n - 1) * Qnm1
        term3 = h(n - 2) * Qnm2
        numerator = term1 - term2 - term3
        denominator = f(n)
        return numerator / denominator


@lru_cache()
def g(n_minus_1):
    """g(m-1) from oe-18-19-19700 eq. (A.15)"""
    if n_minus_1 == 0:
        return - 1 / 2
    else:
        n_minus_2 = n_minus_1 - 1
        return - (1 + g(n_minus_2) * h(n_minus_2)) / f(n_minus_1)


@lru_cache()
def h(n_minus_2):
    """h(m-2) from oe-18-19-19700 eq. (A.14)"""
    n = n_minus_2 + 2
    return -n * (n - 1) / (2 * f(n_minus_2))


@lru_cache()
def f(n):
    """f(m) from oe-18-19-19700 eq. (A.16)"""
    if n == 0:
        return 2
    elif n == 1:
        return e.sqrt(19) / 2
    else:
        term1 = n * (n + 1) + 3
        term2 = g(n - 1) ** 2
        term3 = h(n - 2) ** 2
        return e.sqrt(term1 - term2 - term3)


class QBFSCache(object):
    """Cache of Qbfs terms evaluated over the unit circle."""
    def __init__(self):
        """Create a new QBFSCache instance."""
        self.Qs = defaultdict(dict)
        self.Ps = defaultdict(dict)
        self.grids = dict()
        self.Pm = None
        self.Pnm2, self.Pnm1 = None, None
        self.Qnm2, self.Qnm1 = None, None

    def get_QBFS(self, m, samples):
        """Get an array of phase values for a given index, and number of samples."""
        try:
            Qm = self.Qs[m][samples]
        except KeyError:
            rho = self.get_grid(samples)
            self.Pm = self.get_PBFS(m, samples)
            if m > 2:
                self.Pnm2 = self.get_PBFS(m - 2, samples)
                self.Pnm1 = self.get_PBFS(m - 1, samples)
                self.Qnm2 = self.get_QBFS(m - 2, samples)
                self.Qnm1 = self.get_QBFS(m - 1, samples)

            Qm = qbfs_recurrence_Q(m, rho, Pn=self.Pm, Pnm1=self.Pnm1, Pnm2=self.Pnm2,
                                   Qnm1=self.Qnm1, Qnm2=self.Qnm2)
            self.Qs[m][samples] = Qm

        return Qm

    def get_PBFS(self, m, samples):
        """Get an array of P values for a given index."""
        try:
            Pm = self.Ps[m][samples]
        except KeyError:
            rho = self.get_grid(samples)
            if m > 2:
                self.Pnm2 = self.get_PBFS(m - 2, samples)
                self.Pnm1 = self.get_PBFS(m - 1, samples)

            Pm = qbfs_recurrence_P(m, rho, Pnm1=self.Pnm1, Pnm2=self.Pnm2)
            self.Ps[m][samples] = Pm

        return Pm

    def get_grid(self, samples):
        """Get a grid of rho coordinates for a given number of samples."""
        try:
            rho = self.grids[samples]
        except KeyError:
            rho, _ = make_rho_phi_grid(samples, aligned='y')
            self.grids[samples] = rho

        return rho

    def __call__(self, number, samples):
        """Get an array of sag values for a given index, norm, and number of samples."""
        return self.get_QBFS(number, samples)

    def clear(self, *args):
        """Empty the cache."""
        self.Qs = defaultdict(dict)
        self.Ps = defaultdict(dict)
        self.grids = dict()


QBFScache = QBFSCache()
config.chbackend_observers.append(QBFScache.clear)


class QBFSSag(Pupil):
    _name = 'Qbfs'
    _cache = QBFScache
    """Qbfs aspheric surface sag, excluding base sphere."""

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
        self : `QBFSSag`
            this QBFSSag instance`

        """
        self.phase = e.zeros([self.samples, self.samples], dtype=config.precision)
        ordered_terms = sorted(self.coefs)
        ordered_values = [self.coefs[v] for v in ordered_terms]
        for term, coef in zip(ordered_terms, ordered_values):
            if coef == 0:
                continue
            else:
                self.phase += self._cache(term, self.samples)
