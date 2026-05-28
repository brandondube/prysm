"""Metadata helpers for the analysis / launch / paraxial / plotting layer.

A LensData carries the system metadata (epd, wavelengths, n_ambient, stop
index).  These helpers let the consuming functions default those values from a
LensData when the caller omits them.

Duck-typed on purpose (no LensData import) so the layer modules can use them
without an import cycle: a LensData exposes a wavelength resolver method, an epd
attribute, etc.; a list of Surface does not.
"""


def lensdata_wavelength(prescription, wvl):
    """Resolve a wavelength against a prescription's metadata.

    When prescription is a LensData, defers to its wavelength resolver
    (None -> reference wavelength, a name string -> microns, a scalar ->
    float).  Otherwise an explicit wavelength is required.
    """
    resolver = getattr(prescription, 'wavelength', None)
    if callable(resolver):
        return float(resolver(wvl))
    if wvl is None:
        raise TypeError('wavelength is required for prescriptions without metadata')
    return float(wvl)


def lensdata_epd(prescription, epd):
    """Entrance pupil diameter, defaulting from a LensData when epd is None.

    Returns None when neither an explicit epd nor a LensData epd is available;
    callers decide whether that is an error.
    """
    if epd is not None:
        return float(epd)
    epd = getattr(prescription, 'epd', None)
    return None if epd is None else float(epd)


def lensdata_stop_index(prescription, stop_index):
    """Aperture-stop index, defaulting from a LensData when stop_index is None."""
    if stop_index is not None:
        return stop_index
    return getattr(prescription, 'stop_index', None)


def lensdata_n_ambient(prescription, n_ambient):
    """Object-space index, defaulting from a LensData when omitted."""
    if n_ambient is not None:
        return float(n_ambient)
    return float(getattr(prescription, 'n_ambient', 1.0))
