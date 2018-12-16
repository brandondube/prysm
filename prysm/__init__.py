"""prysm, a python optics module."""

from prysm.conf import config
from prysm.convolution import Convolvable
from prysm.extras import plot_fourier_chain
from prysm.detector import Detector, OLPF, PixelAperture
from prysm.pupil import Pupil
from prysm.fringezernike import FringeZernike
from prysm.standardzernike import StandardZernike
from prysm.seidel import Seidel
from prysm.psf import PSF, AiryDisk
from prysm.otf import MTF
from prysm.interferogram import Interferogram
from prysm.geometry import (
    gaussian,
    rotated_ellipse,
    square,
    regular_polygon,
    pentagon,
    hexagon,
    heptagon,
    octagon,
    nonagon,
    decagon,
    hendecagon,
    dodecagon,
    trisdecagon
)
from prysm.objects import (
    Slit,
    Pinhole,
    SiemensStar,
    TiltedSquare,
    SlantedEdge,
)

from prysm.lens import Lens

__all__ = [
    'config',
    'plot_fourier_chain',
    'Detector',
    'OLPF',
    'PixelAperture',
    'Pupil',
    'FringeZernike',
    'StandardZernike',
    'Interferogram',
    'Seidel',
    'PSF',
    'AiryDisk',
    'MTF',
    'Lens',
    'gaussian',
    'rotated_ellipse',
    'regular_polygon',
    'square',
    'pentagon',
    'hexagon',
    'heptagon',
    'octagon',
    'nonagon',
    'decagon',
    'hendecagon',
    'dodecagon',
    'trisdecagon',
    'Slit',
    'Pinhole',
    'SiemensStar',
    'TiltedSquare',
    'SlantedEdge',
    'Convolvable',
]
