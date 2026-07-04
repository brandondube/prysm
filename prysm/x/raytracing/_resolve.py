"""System-entry metadata resolvers."""

from ._meta import object_space_index, image_space_index


def compiled_surfaces(system):
    """Return a compiled Surface list for a system or bare sequence."""
    to_surfaces = getattr(system, 'to_surfaces', None)
    if callable(to_surfaces):
        return to_surfaces()
    return list(system)


def resolve_wavelength(system, wavelength):
    """Resolve wavelength, using the system reference when available."""
    resolver = getattr(system, 'wavelength', None)
    if callable(resolver):
        return float(resolver(wavelength))
    if wavelength is None:
        raise ValueError(
            'wavelength must be given for a bare surface sequence; only an '
            'OpticalSystem resolves a None wavelength to its reference.')
    return float(wavelength)


class TraceContext:
    """Compiled surfaces and trace metadata."""

    __slots__ = ('surfaces', 'wavelength', 'epd', 'stop_index',
                 '_n_object', '_n_image')

    def __init__(self, surfaces, wavelength, epd=None, stop_index=None):
        self.surfaces = surfaces
        self.wavelength = float(wavelength)
        self.epd = None if epd is None else float(epd)
        self.stop_index = None if stop_index is None else int(stop_index)
        self._n_object = None
        self._n_image = None

    @property
    def n_object(self):
        """Object-space medium index."""
        if self._n_object is None:
            self._n_object = object_space_index(self.surfaces, self.wavelength)
        return self._n_object

    @property
    def n_image(self):
        """Image-space medium index, falling back to the object side."""
        if self._n_image is None:
            self._n_image = image_space_index(self.surfaces, self.wavelength,
                                              fallback=self.n_object)
        return self._n_image


def trace_context(system, wavelength=None, *, chief=False, epd=None,
                  stop_index=None):
    """Resolve a system or bare sequence into a TraceContext.

    Parameters
    ----------
    system : OpticalSystem, LensData, or sequence of Surface
    wavelength : float, optional
        in microns.
    chief : bool, optional
        also resolve epd and stop_index.
    epd : float, optional
        explicit entrance-pupil diameter; wins over the system value.
    stop_index : int, optional
        explicit aperture-stop index; wins over the system value.

    Returns
    -------
    TraceContext

    """
    surfaces = compiled_surfaces(system)
    wvl = resolve_wavelength(system, wavelength)
    if chief:
        if epd is None:
            resolver = getattr(system, 'entrance_pupil_diameter', None)
            if callable(resolver):
                epd = resolver(wvl)
        if stop_index is None:
            stop_index = getattr(system, 'stop_index', None)
    return TraceContext(surfaces, wvl, epd=epd, stop_index=stop_index)
