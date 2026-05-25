"""First-order optics equations for system modeling."""

from .mathops import np


def object_to_image_dist(efl, object_distance):
    """Compute the image distance from the object distance.

    Parameters
    ----------
    efl : float
        focal length of the lens
    object_distance : float or ndarray
        distance from the object to the front principal plane of the lens,
        negative for an object to the left of the lens

    Returns
    -------
    float
        image distance.  Distance from rear principal plane (assumed to be in
        contact with front principal plane) to image.

    Notes
    -----
    efl and object distance should be in the same units.  Return value will
    be in the same units as the inputs.

    """
    ret = 1 / efl + 1 / object_distance
    return 1 / ret


def image_to_object_dist(efl, image_distance):
    """Compute the object distance from the image distance.

    Parameters
    ----------
    efl : float
        focal length of the lens
    image_distance : float or ndarray
        distance from the object to the front principal plane of the lens,
        positive for an object in front of a lens of positive focal length.

    Notes
    -----
    efl and image distance should be in the same units.  Return value will
    be in the same units as the input.

    """
    ret = 1 / efl - 1 / image_distance
    return 1 / ret


def object_image_to_efl(object_distance, image_distance):
    """Compute focal length from a pair of conjugate distances.

    Parameters
    ----------
    object_distance : float or ndarray
        signed object distance from the front principal plane
    image_distance : float or ndarray
        signed image distance from the rear principal plane

    Returns
    -------
    float or ndarray
        focal length, in the same units as the inputs

    """
    power = 1 / image_distance - 1 / object_distance
    return 1 / power


def efl_to_power(efl, n=1):
    """Convert effective focal length to optical power.

    Parameters
    ----------
    efl : float or ndarray
        effective focal length
    n : float, optional
        refractive index of the surrounding medium

    Returns
    -------
    float or ndarray
        optical power, in inverse units of efl

    """
    return n / efl


def power_to_efl(power, n=1):
    """Convert optical power to effective focal length.

    Parameters
    ----------
    power : float or ndarray
        optical power
    n : float, optional
        refractive index of the surrounding medium

    Returns
    -------
    float or ndarray
        effective focal length

    """
    return n / power


def efl_to_fno(efl, epd):
    """Compute f/# from effective focal length and entrance pupil diameter.

    Parameters
    ----------
    efl : float or ndarray
        effective focal length
    epd : float
        entrance pupil diameter

    Returns
    -------
    float or ndarray
        f/number

    """
    return abs(efl) / epd


def fno_to_efl(fno, epd):
    """Compute effective focal length from f/# and entrance pupil diameter.

    Parameters
    ----------
    fno : float or ndarray
        f/number
    epd : float
        entrance pupil diameter

    Returns
    -------
    float or ndarray
        effective focal length

    """
    return fno * epd


def fno_to_epd(fno, efl):
    """Compute entrance pupil diameter from f/# and effective focal length.

    Parameters
    ----------
    fno : float or ndarray
        f/number
    efl : float
        effective focal length

    Returns
    -------
    float or ndarray
        entrance pupil diameter

    """
    return abs(efl) / fno


def image_dist_epd_to_na(image_distance, epd):
    """Compute the NA from an image distance and entrance pupil diameter.

    Parameters
    ----------
    image_distance : float
        distance from the image to the entrance pupil
    epd : float
        diameter of the entrance pupil

    Returns
    -------
    float
        numerical aperture.  The NA of the system.

    """
    rho = epd / 2
    marginal_ray_angle = abs(np.arctan2(rho, image_distance))
    return marginal_ray_angle


def image_dist_epd_to_fno(image_distance, epd):
    """Compute the f/# from an image distance and entrance pupil diameter.

    Parameters
    ----------
    image_distance : float
        distance from the image to the entrance pupil
    epd : float
        diameter of the entrance pupil

    Returns
    -------
    float
        fno.  The working f/# of the system.

    """
    na = image_dist_epd_to_na(image_distance, epd)
    return na_to_fno(na)


def fno_to_na(fno):
    """Convert an fno to an NA.

    Parameters
    ----------
    fno : float
        focal ratio

    Returns
    -------
    float
        NA.  The NA of the system.

    """
    return 1 / (2 * fno)


def na_to_fno(na):
    """Convert an NA to an f/#.

    Parameters
    ----------
    na : float
        numerical aperture

    Returns
    -------
    float
        fno.  The f/# of the system.

    """
    return 1 / (2 * np.sin(na))


def object_dist_to_mag(efl, object_dist):
    """Compute the linear magnification from the object distance and focal length.

    Parameters
    ----------
    efl : float
        focal length of the lens
    object_dist : float
        object distance

    Returns
    -------
    float
        linear magnification.  Also known as the lateral magnification

    """
    return efl / (efl - object_dist)


def mag_to_object_dist(efl, mag):
    """Compute the object distance for a given focal length and magnification.

    Parameters
    ----------
    efl : float
        focal length of the lens
    mag : float
        signed magnification

    Returns
    -------
    float
        object distance

    """
    return efl * (1 - 1/mag)


def mag_to_image_dist(efl, mag):
    """Compute the image distance for a given focal length and magnification.

    Parameters
    ----------
    efl : float
        focal length of the lens
    mag : float
        signed magnification

    Returns
    -------
    float
        image distance

    """
    return efl * (1 - mag)


def linear_to_long_mag(lateral_mag):
    """Compute the longitudinal (along optical axis) magnification from the lateral mag.

    Parameters
    ----------
    lateral_mag : float
        linear magnification, from thin lens formulas

    Returns
    -------
    float
        longitudinal magnification

    """
    return lateral_mag**2


def mag_to_fno(mag, infinite_fno, pupil_mag=1):
    """Compute the working f/# from the magnification and infinite f/#.

    Parameters
    ----------
    mag : float or ndarray
        linear or lateral magnification
    infinite_fno : float
        f/# as defined by EFL/EPD
    pupil_mag : float
        pupil magnification

    Returns
    -------
    float
        working f/number

    """
    return (1 + abs(mag) / pupil_mag) * infinite_fno


def defocus_to_image_displacement(W020, fno, wavelength=None):
    """Compute image displacment from wavefront defocus expressed in waves 0-P to.

    Parameters
    ----------
    W020 : float or ndarray
        wavefront defocus, units of waves if wavelength != None, else units of length
    fno : float
        f/# of the lens or system
    wavelength : float, optional
        wavelength of light, if None W020 takes units of length

    Returns
    -------
    float
        image displacement.  Motion of image in um caused by defocus OPD

    """
    if wavelength is not None:
        return 8 * fno**2 * wavelength * W020
    else:
        return 8 * fno**2 * W020


def image_displacement_to_defocus(dz, fno, wavelength=None):
    """Compute the wavefront defocus from image shift, expressed in the same units as the shift.

    Parameters
    ----------
    dz : float or ndarray
        displacement of the image
    fno : float
        f/# of the lens or system
    wavelength : float, optional
        wavelength of light, if None return has units the same as dz, else waves

    Returns
    -------
    float
        wavefront defocus, waves if Wavelength != None, else same units as dz

    """
    if wavelength is not None:
        return dz / (8 * fno ** 2 * wavelength)
    else:
        return dz / (8 * fno ** 2)


def image_shift_to_tilt(dx, fno):
    """Compute the wavefront tilt associated with an image shift.

    Parameters
    ----------
    dx : float or ndarray
        translation of the image
    fno : float
        f/# of the lens or system

    Returns
    -------
    float
        wavefront tilt W111, same units as dx
        W111 has a peak-to-valley of 2, and "amplitude" of 1
        to convert to Z2 or Z3, those have a peak-to-valley of 4, so
        divide by two for amplitude coefficients, or 4 for RMS coefficients

    """
    return (dx/fno)*0.5


def tilt_to_image_shift(W111, fno):
    """Compute image shift from wavefront tilt.

    Parameters
    ----------
    W111 : float or ndarray
        wavefront tilt, unit amplitude (peak-to-valley of 2)
    fno : float
        f/# of the lens or system

    Returns
    -------
    float
        image translation, in same units as W111 (e.g., um)

    """
    return 2*(W111*fno)


def singlet_power(c1, c2, t, n, n_ambient=1.):
    """Optical power of a thick singlet.

    Parameters
    ----------
    c1 : float
        curvature of S1
    c2 : float
        curvature of S2
    t : float
        vertex-to-vertex thickness
    n : float
        refractive index
    n_ambient: float
        refractive index of the ambient medium ("air")

    Returns
    -------
    float
        optical power in the ambient medium

    """
    phi1 = (n - n_ambient) * c1
    phi2 = (n_ambient - n) * c2
    return phi1 + phi2 - t/n * phi1 * phi2


def singlet_efl(c1, c2, t, n, n_ambient=1.):
    """EFL of a singlet.

    Parameters
    ----------
    c1 : float
        curvature of S1
    c2 : float
        curvature of S2
    t : float
        vertex-to-vertex thickness
    n : float
        refractive index
    n_ambient: float
        refractive index of the ambient medium ("air")

    Returns
    -------
    float
        EFL

    """
    phi = singlet_power(c1, c2, t, n, n_ambient)
    return n_ambient / phi


def singlet_bfl(c1, c2, t, n, n_ambient=1.):
    """Back focal length of a thick singlet.

    Parameters
    ----------
    c1 : float
        curvature of S1
    c2 : float
        curvature of S2
    t : float
        vertex-to-vertex thickness
    n : float
        refractive index
    n_ambient: float
        refractive index of the ambient medium ("air")

    Returns
    -------
    float
        signed distance from S2 to the rear focal point

    """
    phi1 = (n - n_ambient) * c1
    efl = singlet_efl(c1, c2, t, n, n_ambient)
    return efl * (1 - t/n * phi1)


def singlet_ffl(c1, c2, t, n, n_ambient=1.):
    """Front focal length of a thick singlet.

    Parameters
    ----------
    c1 : float
        curvature of S1
    c2 : float
        curvature of S2
    t : float
        vertex-to-vertex thickness
    n : float
        refractive index
    n_ambient: float
        refractive index of the ambient medium ("air")

    Returns
    -------
    float
        signed distance from S1 to the front focal point

    """
    phi2 = (n_ambient - n) * c2
    efl = singlet_efl(c1, c2, t, n, n_ambient)
    return -efl * (1 - t/n * phi2)


def twolens_efl(efl1, efl2, separation):
    """Use thick lens equations to compute the focal length for two elements separated by some distance.

    Parameters
    ----------
    efl1 : float
        EFL of the first lens
    efl2 : float
        EFL of the second lens

    separation : float
        separation of the two lenses

    Returns
    -------
    float
        focal length of the two lens system

    """
    phi1, phi2, t = 1 / efl1, 1 / efl2, separation
    phi_tot = phi1 + phi2 - t * phi1 * phi2
    return 1 / phi_tot


def twolens_power(efl1, efl2, separation):
    """Compute the optical power for two thin lenses in air.

    Parameters
    ----------
    efl1 : float
        EFL of the first lens
    efl2 : float
        EFL of the second lens
    separation : float
        separation of the two lenses

    Returns
    -------
    float
        optical power of the two lens system

    """
    return 1 / twolens_efl(efl1, efl2, separation)


def twolens_bfl(efl1, efl2, separation):
    """Use thick lens equations to compute the back focal length for two elements separated by some distance.

    Parameters
    ----------
    efl1 : float
        EFL of the first lens
    efl2 : float
        EFL of the second lens

    separation : float
        separation of the two lenses.

    Returns
    -------
    float
        back focal length of the two lens system.

    """
    phi1 = 1 / efl1
    numerator = 1 - separation * phi1
    efl = twolens_efl(efl1, efl2, separation)
    return numerator * efl


def twolens_ffl(efl1, efl2, separation):
    """Compute the front focal length for two thin lenses in air.

    Parameters
    ----------
    efl1 : float
        EFL of the first lens
    efl2 : float
        EFL of the second lens
    separation : float
        separation of the two lenses

    Returns
    -------
    float
        front focal length of the two lens system

    """
    phi2 = 1 / efl2
    efl = twolens_efl(efl1, efl2, separation)
    return -efl * (1 - separation * phi2)


def twolens_separation(efl1, efl2, efl):
    """Compute the separation required for a target two-lens EFL.

    Parameters
    ----------
    efl1 : float
        EFL of the first lens
    efl2 : float
        EFL of the second lens
    efl : float
        target EFL of the two-lens system

    Returns
    -------
    float
        separation between the two lenses

    """
    phi1 = 1 / efl1
    phi2 = 1 / efl2
    phi = 1 / efl
    return (phi1 + phi2 - phi) / (phi1 * phi2)
