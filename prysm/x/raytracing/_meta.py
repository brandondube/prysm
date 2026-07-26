"""Object/image-space index helpers over compiled surfaces."""

from .spencer_and_murty import STYPE_REFRACT, _is_measurement_surf


def _surface_medium_index(surface, wavelength, fallback):
    """Evaluate a surface's post-surface medium index, or return fallback."""
    material = getattr(surface, 'material', None)
    if material is not None:
        return float(material.n(wavelength))
    return float(fallback)


def object_space_index(surfaces, wavelength):
    """Resolve the object-space medium index from the object surface.

    If there is no leading measurement surface, object space is air.
    """
    if (len(surfaces) > 0
            and _is_measurement_surf(getattr(surfaces[0], 'typ', None))):
        return _surface_medium_index(surfaces[0], wavelength, 1.0)
    return 1.0


def object_image_indices(surfaces, wavelength):
    """Resolve (n_object, n_image), the image falling back to the object."""
    n_object = object_space_index(surfaces, wavelength)
    n_image = image_space_index(surfaces, wavelength, fallback=n_object)
    return n_object, n_image


def image_space_index(surfaces, wavelength, fallback=1.0):
    """Resolve the image-space medium index from an explicit image surface.

    A bare sequence ending at a powered surface has no explicit image plane.
    """
    if len(surfaces) == 0:
        return float(fallback)
    if not _is_measurement_surf(getattr(surfaces[-1], 'typ', None)):
        raise ValueError(
            'image-space index requires a trailing eval image surface; append '
            'an explicit image surface instead of relying on a bare final '
            'powered surface.'
        )
    n = object_space_index(surfaces, wavelength)
    if len(surfaces) == 1:
        return n
    # Walk the actual medium transitions.  EVAL planes and mirrors do not
    # change the medium, so any number of them may precede IMAGE.
    start = 1 if _is_measurement_surf(
        getattr(surfaces[0], 'typ', None)
    ) else 0
    for surface in surfaces[start:]:
        if getattr(surface, 'typ', None) == STYPE_REFRACT:
            n = _surface_medium_index(surface, wavelength, n)
    return float(n)
