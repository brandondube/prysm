"""A collection of thin lens equations for system modeling."""

from .mathops import engine as e
from .util import guarantee_array
from .zernike import defocus as _defocus


def object_to_image_dist(efl, object_distance):
    """Compute the image distance from the object distance.

    Parameters
    ----------
    efl : `float`
        focal length of the lens
    object_distance : `float` or `numpy.ndarray`
        distance from the object to the front principal plane of the lens,
        negative for an object to the left of the lens

    Returns
    -------
    `float`
        image distance.  Distance from rear principal plane (assumed to be in
        contact with front principal plane) to image.

    Notes
    -----
    efl and object distance should be in the same units.  Return value will
    be in the same units as the inputs.

    """
    object_distance = guarantee_array(object_distance)
    ret = 1 / efl + 1 / object_distance
    return 1 / ret


def image_to_object_dist(efl, image_distance):
    """Compute the object distance from the image distance.

    Parameters
    ----------
    efl : `float`
        focal length of the lens
    object_distance : `float` or `numpy.ndarray`
        distance from the object to the front principal plane of the lens,
        positive for an object in front of a lens of positive focal length.

    Notes
    -----
    efl and image distance should be in the same units.  Return value will
    be in the same units as the input.

    """
    image_distance = guarantee_array(image_distance)
    ret = 1 / efl - 1 / image_distance
    return 1 / ret


def image_dist_epd_to_na(image_distance, epd):
    """Compute the NA from an image distance and entrance pupil diameter.

    Parameters
    ----------
    image_distance : `float`
        distance from the image to the entrance pupil
    epd : `float`
        diameter of the entrance pupil

    Returns
    -------
    `float`
        numerical aperture.  The NA of the system.

    """
    image_distance = guarantee_array(image_distance)

    rho = epd / 2
    marginal_ray_angle = abs(e.arctan2(rho, image_distance))
    return marginal_ray_angle


def image_dist_epd_to_fno(image_distance, epd):
    """Compute the f/# from an image distance and entrance pupil diameter.

    Parameters
    ----------
    image_distance : `float`
        distance from the image to the entrance pupil
    epd : `float`
        diameter of the entrance pupil

    Returns
    -------
    `float`
        fno.  The working f/# of the system.

    """
    na = image_dist_epd_to_na(image_distance, epd)
    return na_to_fno(na)


def fno_to_na(fno):
    """Convert an fno to an NA.

    Parameters
    ----------
    fno : `float`
        focal ratio

    Returns
    -------
    `float`
        NA.  The NA of the system.

    """
    return 1 / (2 * fno)


def na_to_fno(na):
    """Convert an NA to an f/#.

    Parameters
    ----------
    na : `float`
        numerical aperture

    Returns
    -------
    `float`
        fno.  The f/# of the system.

    """
    return 1 / (2 * e.sin(na))


def object_dist_to_mag(efl, object_dist):
    """Compute the linear magnification from the object distance and focal length.

    Parameters
    ----------
    efl : `float`
        focal length of the lens
    object_dist : `float`
        object distance

    Returns
    -------
    `float`
        linear magnification.  Also known as the lateral magnification

    """
    object_dist = guarantee_array(object_dist)
    return efl / (efl - object_dist)


def mag_to_object_dist(efl, mag):
    """Compute the object distance for a given focal length and magnification.

    Parameters
    ----------
    efl : `float`
        focal length of the lens
    mag : `float`
        signed magnification

    Returns
    -------
    `float`
        object distance

    """
    return efl * ((1/mag) + 1)


def linear_to_long_mag(lateral_mag):
    """Compute the longitudinal (along optical axis) magnification from the lateral mag.

    Parameters
    ----------
    lateral_mag : `float`
        linear magnification, from thin lens formulas

    Returns
    -------
    `float`
        longitudinal magnification

    """
    return lateral_mag**2


def mag_to_fno(mag, infinite_fno, pupil_mag=1):
    """Compute the working f/# from the magnification and infinite f/#.

    Parameters
    ----------
    mag : `float` or `numpy.ndarray`
        linear or lateral magnification
    infinite_fno : `float`
        f/# as defined by EFL/EPD
    pupil_mag : `float`
        pupil magnification

    Returns
    -------
    `float`
        working f/number

    """
    mag = guarantee_array(mag)
    return (1 + abs(mag) / pupil_mag) * infinite_fno


def defocus_to_image_displacement(defocus, fno, wavelength, zernike=False, norm=False):
    """Compute image displacment from wavefront defocus expressed in waves 0-P to.

    Parameters
    ----------
    defocus : `float` or `numpy.ndarray`
        wavefront defocus
    fno : `float`
        f/# of the lens or system
    wavelength : `float`
        wavelength of light, expressed in micron
    zernike : `bool`
        zernike model of defocus (otherwise model is Seidel)
    norm : `bool`
        if zernike model, term is rms normalized

    Returns
    -------
    `float`
        image displacement.  Motion of image in um caused by defocus OPD

    """
    defocus = guarantee_array(defocus)

    # if the defocus is a zernike, make it match Seidel notation for equation validity
    if zernike is True:
        if norm is True:
            defocus = defocus * _defocus.norm  # not using *= on these to avoid side effects with in-place ops
        defocus = defocus * 2
    return 8 * fno**2 * wavelength * defocus


def image_displacement_to_defocus(image_displacement, fno, wavelength, zernike=False, norm=False):
    """Compute the wavefront defocus from image shift, expressed in the same units as the shift.

    Parameters
    ----------
    image_displacement : `float` or ~`numpy.ndarray`
        displacement of the image
    fno : `float`
        f/# of the lens or system
    wavelength : `float`
        wavelength of light, expressed in microns
    zernike : `bool`
        return in Zernike notation
    norm : `bool`
        subset of zernike -- return rms normalized zernike

    Returns
    -------
    `float`
        wavefront defocus

    """
    image_displacement = guarantee_array(image_displacement)
    defocus = image_displacement / (8 * fno ** 2 * wavelength)
    if zernike is True:
        if norm is True:
            return defocus / 2 / _defocus.norm
        else:
            return defocus / 2
    else:
        return defocus


def twolens_efl(efl1, efl2, separation):
    """Use thick lens equations to compute the focal length for two elements separated by some distance.

    Parameters
    ----------
    efl1 : `float`
        EFL of the first lens
    efl2 : `float`
        EFL of the second lens

    separation : `float`
        separation of the two lenses

    Returns
    -------
    `float`
        focal length of the two lens system

    """
    phi1, phi2, t = 1 / efl1, 1 / efl2, separation
    phi_tot = phi1 + phi2 - t * phi1 * phi2
    return 1 / phi_tot


def twolens_bfl(efl1, efl2, separation):
    """Use thick lens equations to compute the back focal length for two elements separated by some distance.

    Parameters
    ----------
    efl1 : `float`
        EFL of the first lens
    efl2 : `float`
        EFL of the second lens

    separation : `float`
        separation of the two lenses.

    Returns
    -------
    `float`
        back focal length of the two lens system.

    """
    phi1 = 1 / efl1
    numerator = 1 - separation * phi1
    denomenator = twolens_efl(efl1, efl2, separation)
    return numerator / denomenator
