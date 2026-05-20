"""Shared prescription-reader result types."""


class PrescriptionFile:
    """A parsed sequential prescription file."""

    __slots__ = ('surfaces', 'wavelengths', 'epd', 'stop_index',
                 'fields', 'unit', 'source_path', 'source_format',
                 'extras')

    def __init__(self, surfaces, wavelengths=(), epd=None, stop_index=None,
                 fields=(), unit=None, source_path=None,
                 source_format=None, extras=None):
        self.surfaces = list(surfaces)
        self.wavelengths = list(wavelengths)
        self.epd = epd
        self.stop_index = stop_index
        self.fields = list(fields)
        self.unit = unit
        self.source_path = source_path
        self.source_format = source_format
        self.extras = dict(extras) if extras else {}

    def __repr__(self):
        return (
            f'PrescriptionFile(n_surfaces={len(self.surfaces)}, '
            f'epd={self.epd}, stop={self.stop_index}, '
            f'wavelengths={self.wavelengths}, '
            f"format={self.source_format!r})"
        )
