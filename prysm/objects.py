"""Objects for image simulation with."""

from .conf import config
from .mathops import np, jinc
from .coordinates import optimize_xy_separable


def slit(x, y, width_x, width_y=None):
    """Rasterize a slit or pair of crossed slits.

    Parameters
    ----------
    x : numpy.ndarray
        x coordinates, 1D or 2D
    y : numpy.ndarray
        y coordinates, 1D or 2D
    width_x : float
        the half-width of the slit in x, diameter will be 2x width_x.
        produces a line along the y axis, use None to not do so
    width_y : float
        the half-height of the slit in y, diameter will be 2x width_y.
        produces a line along the y axis, use None to not do so
    orientation : string, {'Horizontal', 'Vertical', 'Crossed', 'Both'}
        the orientation of the slit; Crossed and Both produce the same results

    Notes
    -----
    Default of 0 samples allows quick creation for convolutions without
    generating the image; use samples > 0 for an actual image.

    """
    x, y = optimize_xy_separable(x, y)
    mask = np.zeros((y.size, x.size), dtype=np.bool)
    if width_x is not None:
        wx = width_x / 2
        mask |= abs(x) <= wx
    if width_y is not None:
        wy = width_y / 2
        mask |= abs(y) <= wy

    return mask


def slit_ft(width_x, width_y, fx, fy):
    """Analytic fourier transform of a slit.

    Parameters
    ----------
    width_x : float
        x width of the slit, pass zero if the slit only has width in y
    width_y : float
        y width of the slit, pass zero if the slit only has width in x
    fx : numpy.ndarray
        sample points in x frequency axis
    fy : numpy.ndarray
        sample points in y frequency axis

    Returns
    -------
    numpy.ndarray
        2D array containing the analytic fourier transform

    """
    if width_x is not None and width_y is not None:
        return (np.sinc(fx * width_x) +
                np.sinc(fy * width_y)).astype(config.precision)
    elif width_x is not None and width_y is None:
        return np.sinc(fx * width_x).astype(config.precision)
    else:
        return np.sinc(fy * width_y).astype(config.precision)


def pinhole(radius, rho):
    """Rasterize a pinhole.

    Parameters
    ----------
    radius : float
        radius of the pinhole
    rho : numpy.ndarray
        radial coordinates

    Returns
    -------
    numpy.ndarray
        2D array containing the pinhole

    """
    return rho <= radius


def pinhole_ft(radius, fr):
    """Analytic fourier transform of a pinhole.

    Parameters
    ----------
    radius : float
        radius of the pinhole
    fr : numpy.ndarray
        radial spatial frequency

    Returns
    -------
    numpy.ndarray
        2D array containing the analytic fourier transform

    """
    fr2 = fr * (radius * 2 * np.pi)
    return jinc(fr2)


def siemensstar(r, t, spokes, oradius=0.9, iradius=0, background='black', contrast=0.9, sinusoidal=False):
    """Rasterize a Siemen's Star.

    Parameters
    ----------
    r : numpy.ndarray
        radial coordinates, 2D
    t : numpy.ndarray
        azimuthal coordinates, 2D
    spokes : int
        number of spokes in the star
    oradius : float
        outer radius of the star
    iradius : float
        inner radius of the star
    background : str, optional, {'black', 'white'}
        background color
    contrast : float, optional
        contrast of the star, 1 = perfect black/white
    sinusoidal : bool, optional
        if True, generates a sinusoidal Siemen' star, else, generates a bar/block siemen's star

    Returns
    -------
    numpy.ndarray
        2D array of the same shape as r, t which is in the range [0,1]

    """
    background = background.lower()
    delta = (1 - contrast)/2
    bottom = delta
    top = 1 - delta
    # generate the siemen's star as a (rho,phi) polynomial
    arr = contrast * np.cos(spokes / 2 * t)

    # scale to (0,1) and clip into a disk
    arr = (arr + 1) / 2
    mask = r > oradius
    mask |= r < iradius

    if background in ('b', 'black'):
        arr[mask] = 0
    elif background in ('w', 'white'):
        arr[mask] = 1
    else:
        raise ValueError('invalid background color')

    if not sinusoidal:  # make binary
        arr[arr < 0.5] = bottom
        arr[arr > 0.5] = top

    return arr


def tiltedsquare(x, y, angle=4, radius=0.5, contrast=0.9, background='white'):
    """Rasterize a tilted square.

    Parameters
    ----------
    x : numpy.ndarray
        x coordinates, 2D
    y : numpy.ndarray
        y coordinates, 2D
    angle : float
        counter-clockwise angle of the square from x, degrees
    radius : float
        radius of the square
    contrast : float
        contrast of the square
    background: str, optional, {'white', 'black'}
        whether to paint a white square on a black background or vice-versa

    Returns
    -------
    numpy.ndarray
        ndarray containing the rasterized square

    """
    background = background.lower()
    delta = (1 - contrast) / 2

    angle = np.radians(angle)
    xp = x * np.cos(angle) - y * np.sin(angle)
    yp = x * np.sin(angle) + y * np.cos(angle)
    mask = (abs(xp) <= radius) * (abs(yp) <= radius)

    arr = np.zeros_like(x)
    if background in ('b', 'white'):
        arr[~mask] = (1 - delta)
        arr[mask] = delta
    else:
        arr[~mask] = delta
        arr[mask] = (1 - delta)

    return arr


def slantededge(x, y, angle=4, contrast=0.9, crossed=False):
    """Rasterize a slanted edge.

    Parameters
    ----------
    x : numpy.ndarray
        x coordinates, 2D
    y : numpy.ndarray
        y coordinates, 2D
    angle : float
        angle of the edge to the cartesian y axis
    contrast : float
        contrast of the edge
    crossed : bool, optional
        if True, draw crossed edges instead of just one

    """

    diff = (1 - contrast) / 2
    arr = np.full(x.shape, 1 - diff)

    angle = np.radians(angle)
    xp = x * np.cos(angle) - y * np.sin(angle)
    mask = xp > 0  # single edge
    if crossed:
        mask = xp > 0  # set of 4 edges
        upperright = mask & np.rot90(mask)
        lowerleft = np.rot90(upperright, 2)
        mask = upperright | lowerleft

    arr[mask] = diff

    return arr
