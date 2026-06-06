"""Rugate / inhomogeneous-index synthesis.

A rugate filter is a coating with a continuous refractive-index profile n(z)
rather than discrete homogeneous layers; a sinusoidal n(z) produces a narrow
reflectance notch (rejection band) and -- being smooth -- avoids the higher-order
harmonics of a discrete quarter-wave stack.  The package's transfer-matrix engine
already handles a stack of many thin homogeneous sublayers and its O(N) gradient
scales to hundreds of them, so a continuous profile is designed here and then
discretized into such a stack, optionally refined index-by-index against the
exact spectrum (refine(..., variables='index')).

Two synthesis routes are provided: a direct parametric sinusoidal rugate, and the
Fourier / Q-function inverse method (Bovard / Sossi), which maps a target
reflectance amplitude versus wavenumber to a log-index profile in optical-
thickness space.  Apodization (a smooth amplitude taper, e.g. a quintic window)
suppresses the sidebands a hard-truncated profile would ring with.

Conventions: thicknesses and wavelengths are microns; wavenumber is k = 2 pi /
lambda; the design notch of a sinusoid of physical period Lambda sits at the
first-order wavelength lambda0 = 2 n_avg Lambda.
"""

from prysm.conf import config
from prysm.mathops import np

from .stack import Stack


# ---------------------------------------------------------------- windows

def quintic_taper(edge_fraction=0.5):
    """A smooth amplitude window that ramps 0 -> 1 -> 0 with a quintic smoothstep.

    The quintic S(t) = 10 t^3 - 15 t^4 + 6 t^5 has zero first and second
    derivatives at both ends, so the modulation switches on and off without the
    kinks that seed reflectance sidebands.

    Parameters
    ----------
    edge_fraction : float
        fraction of the profile over which the window ramps at each end (<= 0.5).

    Returns
    -------
    callable
        w(u) for u in [0, 1].

    """
    e = float(edge_fraction)

    def smoothstep(t):
        t = np.clip(t, 0.0, 1.0)
        return t * t * t * (10 - 15 * t + 6 * t * t)

    def window(u):
        u = np.asarray(u, dtype=config.precision)
        if e <= 0:
            return np.ones_like(u)
        rising = smoothstep(u / e)
        falling = smoothstep((1.0 - u) / e)
        return np.minimum(rising, falling)

    return window


# ---------------------------------------------------------------- discretize

def discretize_profile(n_of_z, total_thickness, n_sublayers, substrate_index,
                       ambient_index=1.0):
    """Sample a continuous index profile into a Stack of thin homogeneous layers.

    The profile is sampled at the midpoint of each of n_sublayers equal slabs of
    thickness total_thickness / n_sublayers.

    Parameters
    ----------
    n_of_z : callable
        index as a function of depth z (microns), z = 0 at the ambient side.
    total_thickness : float
        total physical thickness, microns.
    n_sublayers : int
        number of homogeneous sublayers.
    substrate_index : float, complex, or callable
        index of the medium after the profile.
    ambient_index : float, complex, or callable, optional
        incidence-medium index.

    Returns
    -------
    Stack

    """
    edges = np.linspace(0.0, total_thickness, n_sublayers + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    indices = [n_of_z(float(z)) for z in centers]
    thicknesses = np.full(n_sublayers, total_thickness / n_sublayers,
                          dtype=config.precision)
    return Stack(indices, thicknesses, substrate_index, ambient_index)


# ---------------------------------------------------------------- parametric

def rugate_period(n_avg, design_wvl):
    """Physical period for a first-order rugate notch at design_wvl."""
    return design_wvl / (2.0 * n_avg)


def notch_wavelength(n_avg, period):
    """First-order notch wavelength of a rugate of given period."""
    return 2.0 * n_avg * period


def sinusoidal_rugate(n_avg, n_amp, design_wvl, n_periods, *,
                      sublayers_per_period=30, substrate_index=None,
                      ambient_index=1.0, apodization=None, clamp=None):
    """Build a sinusoidal rugate stack with a notch at design_wvl.

    The index profile is n(z) = n_avg + n_amp * w(z / total) * sin(2 pi z /
    Lambda), with Lambda the first-order period for design_wvl and w an optional
    amplitude window (apodization).

    Parameters
    ----------
    n_avg : float
        mean index.
    n_amp : float
        modulation amplitude (peak-to-peak is 2 n_amp).
    design_wvl : float
        notch wavelength, microns.
    n_periods : int
        number of sinusoidal periods.
    sublayers_per_period : int, optional
        discretization density.
    substrate_index : float, optional
        substrate index; defaults to n_avg (index-matched, isolating the notch).
    ambient_index : float, optional
        incidence-medium index.
    apodization : callable, optional
        amplitude window w(u), u in [0, 1] (e.g. quintic_taper()); default none.
    clamp : (float, float), optional
        clip the profile to this (min, max) index range.

    Returns
    -------
    Stack

    """
    Lambda = rugate_period(n_avg, design_wvl)
    total = n_periods * Lambda
    if substrate_index is None:
        substrate_index = n_avg
    win = apodization

    def n_of_z(z):
        amp = n_amp
        if win is not None:
            amp = n_amp * float(win(z / total))
        n = n_avg + amp * np.sin(2 * np.pi * z / Lambda)
        if clamp is not None:
            n = np.clip(n, clamp[0], clamp[1])
        return n

    n_sub = int(round(n_periods * sublayers_per_period))
    return discretize_profile(n_of_z, total, n_sub, substrate_index, ambient_index)


def apodize(n_of_z, n_avg, total_thickness, window):
    """Wrap a profile so its modulation about n_avg is tapered by a window.

    Returns a new callable n'(z) = n_avg + window(z / total) * (n_of_z(z) -
    n_avg), i.e. the index excursion is scaled by the window while the mean is
    preserved.

    Parameters
    ----------
    n_of_z : callable
        the base index profile.
    n_avg : float
        the mean index the modulation rides on.
    total_thickness : float
        total profile thickness, microns.
    window : callable
        amplitude window w(u), u in [0, 1].

    Returns
    -------
    callable

    """
    def tapered(z):
        return n_avg + float(window(z / total_thickness)) * (n_of_z(z) - n_avg)

    return tapered


# ---------------------------------------------------------------- inverse

def rugate_from_target(wavenumbers, target_amplitude, n_avg,
                       total_optical_thickness, n_sublayers, *,
                       substrate_index=None, ambient_index=1.0, clamp=None):
    """Fourier (Q-function) synthesis of an index profile from a target spectrum.

    The first-order inverse of thin-film reflection: in the optical-thickness
    coordinate x the log-index derivative Q(x) = (1/2) d ln n / dx is the Fourier
    transform of the target reflectance amplitude r(k) (k the wavenumber, the
    factor 2 from the round trip),

        Q(x) = (1 / pi) Re integral r(k) exp(2 i k x) dk,

    so ln n(x) = ln n_avg + 2 integral_0^x Q(x') dx'.  The profile is then mapped
    from optical to physical thickness (dz = dx / n) and discretized.

    Parameters
    ----------
    wavenumbers : ndarray
        k grid, k = 2 pi / lambda (inverse microns).
    target_amplitude : ndarray
        target reflectance amplitude r(k) on that grid (0..1, real).
    n_avg : float
        mean index the profile rides on.
    total_optical_thickness : float
        optical thickness span of the synthesized profile, microns.
    n_sublayers : int
        number of homogeneous sublayers in the output stack.
    substrate_index : float, optional
        substrate index; defaults to n_avg.
    ambient_index : float, optional
        incidence-medium index.
    clamp : (float, float), optional
        clip the profile to this (min, max) index range.

    Returns
    -------
    Stack

    """
    k = np.asarray(wavenumbers, dtype=config.precision)
    r = np.asarray(target_amplitude, dtype=config.precision)
    dk = k[1] - k[0]

    x = np.linspace(0.0, total_optical_thickness, max(n_sublayers * 4, 2000))
    # Q(x) = (1/pi) Re integral r(k) exp(2 i k x) dk
    phase = np.exp(2j * np.outer(x, k))
    Q = (1.0 / np.pi) * np.real((r[None, :] * phase).sum(axis=1)) * dk
    ln_n = np.log(n_avg) + 2.0 * np.cumsum(Q) * (x[1] - x[0])
    n_x = np.exp(ln_n)
    if clamp is not None:
        n_x = np.clip(n_x, clamp[0], clamp[1])

    # map optical thickness x -> physical depth z (dz = dx / n)
    dz = (x[1] - x[0]) / n_x
    z = np.concatenate([np.zeros(1), np.cumsum(dz[:-1])])
    total_z = float(z[-1])

    def n_of_z(zz):
        return float(np.interp(zz, z, n_x))

    if substrate_index is None:
        substrate_index = n_avg
    return discretize_profile(n_of_z, total_z, n_sublayers, substrate_index,
                              ambient_index)


__all__ = [
    'quintic_taper',
    'discretize_profile',
    'rugate_period',
    'notch_wavelength',
    'sinusoidal_rugate',
    'apodize',
    'rugate_from_target',
]
