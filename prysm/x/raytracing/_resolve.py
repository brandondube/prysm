"""System-entry metadata resolvers (ADR-0001).

These accept a system (OpticalSystem or LensData) and resolve metadata through
its own methods, falling back to an explicit-scalar requirement for a bare
surface sequence.  The leaf primitives in _meta stay system-free.
"""


def compiled_surfaces(system):
    """Compiled Surface list for a system/LensData, or the sequence itself.

    Always returns a list: an OpticalSystem or LensData compiles via
    to_surfaces(); a bare surface sequence is copied into a list.
    """
    to_surfaces = getattr(system, 'to_surfaces', None)
    if callable(to_surfaces):
        return to_surfaces()
    return list(system)


def resolve_wavelength(system, wavelength):
    """Resolve a None wavelength via the system; require explicit for a list.

    A system maps None to its reference wavelength; a bare sequence carries no
    metadata, so None is an error there.
    """
    resolver = getattr(system, 'wavelength', None)
    if callable(resolver):
        return float(resolver(wavelength))
    if wavelength is None:
        raise ValueError(
            'wavelength must be given for a bare surface sequence; only an '
            'OpticalSystem resolves a None wavelength to its reference.')
    return float(wavelength)


def resolve_chief_metadata(system, wavelength, epd, stop_index):
    """Compile surfaces and resolve wvl/epd/stop for a chief-ray analysis.

    A system fills None wvl/epd/stop from its metadata; a bare sequence needs
    wavelength given and leaves epd/stop as passed.

    Returns
    -------
    surfaces, wvl, epd, stop_index
    """
    if hasattr(system, 'wavelength') and hasattr(system, 'to_surfaces'):
        wvl = system.wavelength(wavelength)
        epd = system.entrance_pupil_diameter(wvl) if epd is None else float(epd)
        stop_index = (stop_index if stop_index is not None
                      else system.stop_index)
        return system.to_surfaces(), wvl, epd, stop_index
    if wavelength is None:
        raise ValueError(
            'wavelength must be given for a bare surface sequence; only an '
            'OpticalSystem resolves a None wavelength to its reference.')
    return (list(system), float(wavelength),
            None if epd is None else float(epd), stop_index)
