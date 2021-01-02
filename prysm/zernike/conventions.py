"""Conversion between various conventions of Zernike polynomials."""

from prysm.mathops import np

from prysm.util import is_odd, sign


def n_m_to_fringe(n, m):
    """Convert (n,m) two term index to Fringe index."""
    term1 = (1 + (n + abs(m))/2)**2
    term2 = 2 * abs(m)
    term3 = (1 + sign(m)) / 2
    return int(term1 - term2 - term3) + 1  # shift 0 base to 1 base


def n_m_to_ansi_j(n, m):
    """Convert (n,m) two term index to ANSI single term index."""
    return int((n * (n + 2) + m) / 2)


def ansi_j_to_n_m(idx):
    """Convert ANSI single term to (n,m) two-term index."""
    n = int(np.ceil((-3 + np.sqrt(9 + 8*idx))/2))
    m = 2 * idx - n * (n + 2)
    return n, m


def noll_to_n_m(idx):
    """Convert Noll Z to (n, m) two-term index."""
    # I don't really understand this code, the math is inspired by POPPY
    # azimuthal order
    n = int(np.ceil((-1 + np.sqrt(1 + 8 * idx)) / 2) - 1)
    if n == 0:
        m = 0
    else:
        # this is sort of a rising factorial to use that term incorrectly
        nseries = int((n + 1) * (n + 2) / 2)
        res = idx - nseries - 1

        if is_odd(idx):
            sign = -1
        else:
            sign = 1

        if is_odd(n):
            ms = [1, 1]
        else:
            ms = [0]

        for i in range(n // 2):
            ms.append(ms[-1] + 2)
            ms.append(ms[-1])

        m = ms[res] * sign

    return n, m


def fringe_to_n_m(idx):
    """Convert Fringe Z to (n, m) two-term index."""
    m_n = 2 * (np.ceil(np.sqrt(idx)) - 1)  # sum of n+m
    g_s = (m_n / 2)**2 + 1  # start of each group of equal n+m given as idx index
    n = m_n / 2 + np.floor((idx - g_s) / 2)
    m = (m_n - n) * (1 - np.mod(idx-g_s, 2) * 2)
    return int(n), int(m)
