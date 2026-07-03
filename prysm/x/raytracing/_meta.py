"""Object/image-space index helpers, pure over a compiled surface list.

The system-metadata resolvers that once lived here (wavelength, EPD, stop,
first-order) were retired in ADR-0001: metadata resolution lives on
OpticalSystem, and the free numerical functions take already-resolved scalars.
These remaining helpers read the object/image-space medium index directly from
the surface sequence, so they stay pure primitives.
"""

from .spencer_and_murty import _is_measurement_surf


def _surface_medium_index(surface, wavelength, fallback):
    """Evaluate a surface's post-surface medium index, or return fallback."""
    material = getattr(surface, 'material', None)
    if material is not None:
        return float(material.n(wavelength))
    return float(fallback)


def object_space_index(surfaces, wavelength):
    """Resolve the object-space medium index from the object surface.

    When the surfaces begin with an OBJECT (or leading eval) measurement
    plane, that row's material is the object-space medium.  Otherwise the object
    space is air (n = 1.0).
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

    Sequential systems place the image plane as the final IMAGE (or eval)
    measurement surface; the medium immediately before that plane is therefore
    the post-surface medium of the penultimate surface.  A bare surface sequence
    ending at a powered surface has no explicit image plane, so callers that need
    an image-space medium must append a final IMAGE/eval surface.
    """
    if len(surfaces) == 0:
        return float(fallback)
    if not _is_measurement_surf(getattr(surfaces[-1], 'typ', None)):
        raise ValueError(
            'image-space index requires a trailing eval image surface; append '
            'an explicit image surface instead of relying on a bare final '
            'powered surface.'
        )
    if len(surfaces) > 1:
        return _surface_medium_index(surfaces[-2], wavelength, fallback)
    return float(fallback)
