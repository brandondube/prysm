"""Shack-Hartmann phase screens."""
import inspect
from math import ceil

from prysm.coordinates import make_xy_grid
from prysm.segmented import _local_window
from prysm.geometry import rectangle
from prysm.mathops import np, is_odd


def shack_hartmann(pitch, n, efl, wavelength, x, y,
                   aperture=rectangle, aperture_kwargs=None,
                   shift=False):
    """Create the complex screen for a shack hartmann lenslet array.

    Parameters
    ----------
    pitch : float
        lenslet pitch, mm
    n : int or tuple of (int, int)
        number of lenslets
    efl : float
        focal length of each lenslet, mm
    wavelength : float
        wavelength of light, microns
    x : ndarray
        x coordinates that define the space of the lens, mm
    y : ndarray
        y coordinates that define the space of the beam, mm
    aperture : callable, optional
        the aperture can either be:
        f(lenslet_semidiameter, x=x, y=y, **kwargs)
        or
        f(lenslet_semidiameter, r=r, **kwargs)
        typically,  it will be either prysm.geometry.circle or prysm.geometry.rectangle
    aperture_kwargs : dict, optional
        the keyword arguments for the aperture function, if any
    shift : bool, optional
        if True, shift the lenslet array by half a pitch in the +x/+y
        directions

    Returns
    -------
    ndarray
        complex ndarray, such that:
        wf2 = wf * shack_hartmann_complex_screen(... efl=efl)
        wf3 = wf2.free_space(efl=efl)
        wf3 represents the complex E-field at the detector, you are likely
            interested in wf3.intensity

    Notes
    -----
    There are many subtle constraints when simulating Shack-Hartmann sensors:
    1) there must be enough samples across a lenslet to avoid aliasing the phase screen
        i.e., (2pi i / wvl)(r^2 / 2f) evolves slowly; implying that somewhat larger
        F/# lenslets are easier to sample well, or relatively large arrays are required.
        For low-order aberrations at the input in moderate amplitudes, >= 32 samples per
        lenslet is OK, although 64 to 128 or more samples per lenslet should be used for
        beams containing high order aberrations in any meaningful quantity.  For a 64x64
        lenslet array, the lower bound of 32 samples per lenslet = 2048 array
    2) there must be dense enough sampling in the output plane to well sample each point
    spready function, i.e. dx <= (lambda*fno_lenslet)/2
    3) the F/# of the lenslet must be _small_ enough that the lenslets' point spread
    functions only minimally overlap

    """
    if not hasattr(n, '__iter__'):
        n = (n, n)

    if aperture_kwargs is None:
        aperture_kwargs = {}

    sig = inspect.signature(aperture)
    params = sig.parameters
    callxy = 'x' in params and 'y' in params

    dx = x[0, 1] - x[0, 0]
    samples_per_lenslet = int(pitch / dx + 1)  # ensure safe rounding

    xc, yc = make_xy_grid(n, dx=pitch, grid=False)
    if shift:
        if not is_odd(n[0]):
            # even number of lenslets, FFT-aligned make_xy_grid needs positive shift
            xc += (pitch/2)
        if not is_odd(n[1]):
            yc += (pitch/2)

    cx = ceil(x.shape[1]/2)
    cy = ceil(y.shape[0]/2)
    lenslet_rsq = (pitch/2)**2
    total_phase = np.zeros_like(x)

    # naming convention:
    # c = center
    # i,j look indices
    # xx, yy = lenslet center (floating point, not samples)
    # rsq = r^2
    # l = local (local coordinate frame, inside the lenslet window)
    for j, yy in enumerate(yc):
        for i, xx in enumerate(xc):
            win = _local_window(cy, cx, (xx, yy), dx, samples_per_lenslet, x, y)
            lx = x[win] - xx
            ly = y[win] - yy
            rsq = lx * lx + ly * ly
            phase = rsq / (2*efl)
            if callxy:
                phase *= aperture(pitch/2, x=lx, y=ly, **aperture_kwargs)
            else:
                phase *= aperture(lenslet_rsq, r=rsq, **aperture_kwargs)

            total_phase[win] += phase

    prefix = -1j * 2 * np.pi/(wavelength/1e3)
    return np.exp(prefix*total_phase)
