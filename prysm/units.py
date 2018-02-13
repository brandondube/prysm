"""Unit manipulation."""


def waves_to_microns(wavelength):
    """Return a conversion factor to yield microns from OPD expressed in waves.

    Parameters
    ----------
    wavelength : `float`
        wavelength of light, expressed in microns

    Returns
    -------
    `float`
        conversion factor from waves to microns

    """
    return 1 / wavelength


def waves_to_nanometers(wavelength):
    """Return a conversion factor to yield nanometers from OPD expressed in waves.

    Parameters
    ----------
    wavelength : `float`
        wavelength of light, expressed in microns

    Returns
    -------
    `float`
        conversion factor from waves to nanometers

    """
    return 1 / (wavelength * 1e3)


def microns_to_waves(wavelength):
    """Return a conversion factor to yield waves from OPD expressed in microns.

    Parameters
    ----------
    wavelength : `float`
        wavelength of light, expressed in microns

    Returns
    -------
    `float`
        conversion factor from microns to waves

    """
    return wavelength


def nanometers_to_waves(wavelength):
    """Return a conversion factor to yield waves from OPD expressed in nanometers.

    Parameters
    ----------
    wavelength : `float`
        wavelength of light, expressed in microns

    Returns
    -------
    float.
        conversion factor from nanometers to waves

    """
    return wavelength * 1e3
