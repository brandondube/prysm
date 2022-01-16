"""Evaluation routines for point spread functions."""
import numbers
from prysm.fttools import fftrange

from scipy import optimize

from .mathops import (
    np, jinc,
    ndimage,
    special
)
from .coordinates import uniform_cart_to_polar


FIRST_AIRY_ZERO = 1.220
SECOND_AIRY_ZERO = 2.233
THIRD_AIRY_ZERO = 3.238
FIRST_AIRY_ENCIRCLED = 0.8377850436212378
SECOND_AIRY_ENCIRCLED = 0.9099305350850819
THIRD_AIRY_ENCIRCLED = 0.9376474743695488

AIRYDATA = {
    1: (FIRST_AIRY_ZERO, FIRST_AIRY_ENCIRCLED),
    2: (SECOND_AIRY_ZERO, SECOND_AIRY_ENCIRCLED),
    3: (THIRD_AIRY_ZERO, THIRD_AIRY_ENCIRCLED)
}


def estimate_size(data, metric, dx=None, x=None, y=None, criteria='last'):
    """Calculate the "size" of the function in data based on a metric.

    Parameters
    ----------
    data : numpy.ndarray
        f(x,y), 2D
    metric : str or float, {'fwhm', '1/e', '1/e^2', float()}
        what metric to apply
    dx : float
        inter-sample spacing, if x and y != None, they supercede this parameter
    x : numpy.ndarray
        x coordinates, 2D
    y : numpy.ndarray
        y coordinates, 2D
    criteria : str, optional, {'first', 'last'}
        whether to use the first or last occurence of <metric>

    Returns
    -------
    float
        the radial coordinate at which on average the function reaches <metric>

    Raises
    ------
    ValueError
        metric not in ('fwhm', '1/e', '1/e^2', numbers.Number())

    """
    criteria = criteria.lower()
    metric = metric.lower()

    if x is None and y is None:
        y, x = (fftrange(s, dtype=data.dtype)*dx for s in data.shape)

    r, p, polar = uniform_cart_to_polar(x, y, data)
    max_ = polar.max()
    if metric == 'fwhm':
        hm = max_ / 2
    elif metric == '1/e':
        hm = 1 / np.e * max_
    elif metric == '1/e^2':
        hm = 1 / (np.e ** 2) * max_
    elif isinstance(metric, numbers.Number):
        hm = metric
    else:
        raise ValueError('unknown metric, use fwhm, 1/e, or 1/e^2')

    mask = polar > hm

    if criteria == 'first':
        meanidx = np.argmax(mask, axis=1).mean()
        lowidx, remainder = divmod(meanidx, 1)
    elif criteria == 'last':
        meanidx = np.argmax(mask[:, ::-1], axis=1).mean()
        meanidx = mask.shape[1] - meanidx
        lowidx, remainder = divmod(meanidx, 1)
        remainder *= -1  # remainder goes the other way in this case
    else:
        raise ValueError('unknown criteria, use first or last')

    lowidx = int(lowidx)
    return r[lowidx] + remainder * r[1]  # subpixel calculation of r


def fwhm(data, dx=None, x=None, y=None, criteria='last'):
    """Calculate the FWHM of (data).

    Parameters
    ----------
    data : numpy.ndarray
        f(x,y), 2D
    dx : float
        inter-sample spacing, if x and y != None, they supercede this parameter
    x : numpy.ndarray
        x coordinates, 2D
    y : numpy.ndarray
        y coordinates, 2D
    criteria : str, optional, {'first', 'last'}
        whether to use the first or last occurence of <metric>


    Returns
    -------
    float
        the FWHM

    """
    # native calculation is a radius, "HWHM", *2 is FWHM
    return estimate_size(x=x, y=y, dx=dx, data=data, metric='fwhm', criteria=criteria) * 2


def one_over_e(data, dx=None, x=None, y=None, criteria='last'):
    """Calculate the 1/e diameter of data.

    Parameters
    ----------
    data : numpy.ndarray
        f(x,y), 2D
    dx : float
        inter-sample spacing, if x and y != None, they supercede this parameter
    x : numpy.ndarray
        x coordinates, 2D
    y : numpy.ndarray
        y coordinates, 2D
    criteria : str, optional, {'first', 'last'}
        whether to use the first or last occurence of <metric>


    Returns
    -------
    float
        the FWHM

    """
    # native calculation is a radius, "HWHM", *2 is FWHM
    return estimate_size(x=x, y=y, dx=dx, data=data, metric='1/e', criteria=criteria) * 2


def one_over_e_sq(data, dx=None, x=None, y=None, criteria='last'):
    """Calculate the 1/e^2 diameter of data.

    Parameters
    ----------
    data : numpy.ndarray
        f(x,y), 2D
    dx : float
        inter-sample spacing, if x and y != None, they supercede this parameter
    x : numpy.ndarray
        x coordinates, 2D
    y : numpy.ndarray
        y coordinates, 2D
    criteria : str, optional, {'first', 'last'}
        whether to use the first or last occurence of <metric>


    Returns
    -------
    float
        the FWHM

    """
    # native calculation is a radius, "HWHM", *2 is FWHM
    return estimate_size(x=x, y=y, dx=dx, data=data, metric='1/e^2', criteria=criteria) * 2


def centroid(data, dx=None, unit='spatial'):
    """Calculate the centroid of the PSF.

    Parameters
    ----------
    data : numpy.ndarray
        data to centroid
    dx : float
        sample spacing, may be None if unit != spatial
    unit : str, {'spatial', 'pixels'}
        unit to return the centroid in.
        If pixels, corner indexed.  If spatial, center indexed.

    Returns
    -------
    int, int
        if unit == pixels, indices into the array
    float, float
        if unit == spatial, referenced to the origin

    """
    center = (int(np.ceil(c/2)) for c in data.shape)
    com = ndimage.center_of_mass(data)
    if unit != 'spatial':
        return com
    else:
        # tuple - cast from generator
        # sample spacing - indices to units
        # x-c -- index shifted from center
        return tuple(dx * (x-c) for x, c in zip(com, center))


def autocrop(data, px):
    """Crop to a rectangular window around the centroid.

    Parameters
    ----------
    data : numpy.ndarray
        data to crop into
    px : int
        window full width, samples

    Returns
    -------
    numpy.ndarray
        cropped data

    """
    com = centroid(data, unit='pixels')
    cy, cx = (int(c) for c in com)
    w = px // 2
    aoi_y_l = cy - w
    aoi_y_h = aoi_y_l + w
    aoi_x_l = cx - w
    aoi_x_h = aoi_x_l + w
    return data[aoi_y_l:aoi_y_h, aoi_x_l:aoi_x_h]


def airydisk(unit_r, fno, wavelength):
    """Compute the airy disk function over a given spatial distancnp.

    Parameters
    ----------
    unit_r : numpy.ndarray
        ndarray with units of um
    fno : float
        F/# of the system
    wavelength : float
        wavelength of light, um

    Returns
    -------
    numpy.ndarray
        ndarray containing the airy pattern

    """
    u_eff = unit_r * np.pi / wavelength / fno
    return abs(2 * jinc(u_eff)) ** 2


def airydisk_ft(r, fno, wavelength):
    """Compute the Fourier transform of the airy disk.

    Parameters
    ----------
    r : numpy.ndarray
        radial spatial frequency, if wvl has units of um, then r has units of 1/um
    fno : float
        f number of the system, dimensionless
    wavelength : float
        wavelength of light, notionally units of um

    Returns
    -------
    numpy.ndarray
        ndarray of same shape as r

    """
    extinction = 1 / (wavelength * fno)
    s = abs(r) / extinction
    if not isinstance(s, numbers.Number):
        s[s > 1] = 1
    elif s > 1:
        return 0
    return (2 / np.pi) * (np.arccos(s) - s * np.sqrt(1 - s ** 2))


def encircled_energy(psf, dx, radius):
    """Compute the encircled energy of the PSF.

    Parameters
    ----------
    psf : numpy.ndarray
        2D array containing PSF data
    dx : float
        sample spacing of psf
    radius : float or iterable
        radius or radii to evaluate encircled energy at

    Returns
    -------
    encircled energy
        if radius is a float, returns a float, else returns a list.

    Notes
    -----
    implementation of "Simplified Method for Calculating Encircled Energy,"
    Baliga, J. V. and Cohn, B. D., doi: 10.1117/12.944334

    """
    from .otf import mtf_from_psf
    # compute MTF from the PSF
    mtf = mtf_from_psf(psf, dx)
    nx, ny = np.meshgrid(mtf.x, mtf.y)
    nu_p = np.sqrt(nx ** 2 + ny ** 2)
    # this is meaninglessly small and will avoid division by 0
    nu_p[nu_p == 0] = 1e-16
    dnx, dny = ny[1, 0] - ny[0, 0], nx[0, 1] - nx[0, 0]

    if not isinstance(radius, numbers.Number):
        out = []
        for r in radius:
            v = _encircled_energy_core(mtf.data, r / 1e3, nu_p, dnx, dny)
            out.append(v)

        return np.asarray(out)
    else:
        return _encircled_energy_core(mtf.data, radius / 1e3, nu_p, dnx, dny)


def _encircled_energy_core(mtf_data, radius, nu_p, dx, dy):
    """Core computation of encircled energy, based on Baliga 1988.

    Parameters
    ----------
    mtf_data : numpy.ndarray
        unaliased MTF data
    radius : float
        radius of "detector"
    nu_p : numpy.ndarray
        radial spatial frequencies
    dx : float
        x frequency delta
    dy : float
        y frequency delta

    Returns
    -------
    float
        encircled energy for given radius

    """
    integration_fourier = special.j1(2 * np.pi * radius * nu_p) / nu_p
    dat = mtf_data * integration_fourier
    return radius * dat.sum() * dx * dy


def _analytical_encircled_energy(fno, wavelength, points):
    """Compute the analytical encircled energy for a diffraction limited circular aperturnp.

    Parameters
    ----------
    fno : float
        F/#
    wavelength : float
        wavelength of light
    points : numpy.ndarray
        radii of "detector"

    Returns
    -------
    numpy.ndarray
        encircled energy values

    """
    p = points * np.pi / fno / wavelength
    return 1 - special.j0(p)**2 - special.j1(p)**2


def _inverse_analytic_encircled_energy(fno, wavelength, energy=FIRST_AIRY_ENCIRCLED):
    def optfcn(x):
        return (_analytical_encircled_energy(fno, wavelength, x) - energy) ** 2

    return optimize.golden(optfcn)
