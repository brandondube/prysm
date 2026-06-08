"""Metadata helpers for raytracing system wrappers."""

from .spencer_and_murty import STYPE_EVAL


def system_wavelength(prescription, wvl):
    """Resolve a wavelength against a system's metadata.

    When prescription is an OpticalSystem, defers to its wavelength resolver
    (None -> reference wavelength, a name string -> microns, a scalar ->
    float).  For a bare lens / surface sequence, None resolves to the kernel
    default of 0.6328 microns.
    """
    resolver = getattr(prescription, 'wavelength', None)
    if callable(resolver):
        return float(resolver(wvl))
    if wvl is None:
        return 0.6328
    return float(wvl)


def system_epd(prescription, epd, wvl=None):
    """Entrance-pupil diameter, defaulting from a system's aperture spec.

    An explicit epd wins.  Otherwise, when prescription is an OpticalSystem
    carrying an ApertureSpec, the spec's first-order entrance-pupil diameter at
    wvl is used to size the pupil-sampling pattern.  Returns None when neither
    is available; callers decide whether that is an error.
    """
    if epd is not None:
        return float(epd)
    aperture = getattr(prescription, 'aperture', None)
    if aperture is not None:
        return float(aperture.entrance_pupil_diameter(prescription, wvl))
    return None


def system_stop_index(prescription, stop_index):
    """Aperture-stop index, defaulting from a system when stop_index is None."""
    if stop_index is not None:
        return stop_index
    return getattr(prescription, 'stop_index', None)


def _surface_medium_index(surface, wavelength, fallback):
    """Evaluate a surface's post-surface medium index, or return fallback."""
    material = getattr(surface, 'material', None)
    if material is not None:
        return float(material.n(wavelength))
    return float(fallback)


def object_space_index(prescription, wavelength):
    """Resolve the object-space medium index from the object surface.

    When a prescription includes an object surface at index 0 (a leading eval
    row), that row's material is the object-space medium.  Otherwise the object
    space is air (n = 1.0).
    """
    if (len(prescription) > 0
            and getattr(prescription[0], 'typ', None) == STYPE_EVAL):
        return _surface_medium_index(prescription[0], wavelength, 1.0)
    return 1.0


def image_space_index(prescription, wavelength, fallback=1.0):
    """Resolve the image-space medium index from an explicit image surface.

    Sequential systems place the image plane as the final eval surface; the
    medium immediately before that plane is therefore the post-surface medium of
    the penultimate surface.  A bare prescription ending at a powered surface
    has no explicit image plane, so callers that need an image-space medium
    must append a final eval surface.
    """
    if len(prescription) == 0:
        return float(fallback)
    if getattr(prescription[-1], 'typ', None) != STYPE_EVAL:
        raise ValueError(
            'image-space index requires a trailing eval image surface; append '
            'an explicit image surface instead of relying on a bare final '
            'powered surface.'
        )
    if len(prescription) > 1:
        return _surface_medium_index(prescription[-2], wavelength, fallback)
    return float(fallback)
