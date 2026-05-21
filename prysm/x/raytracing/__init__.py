"""Sequential Ray Tracing."""

from .prescription import Prescription

FRAUNHOFER_LINES_UM = {
    'C': 0.6562725,
    'd': 0.5875618,
    'F': 0.4861327,
}

__all__ = [
    'FRAUNHOFER_LINES_UM',
    'Prescription',
]
