"""prysm, a python optics module."""

from prysm.conf import config
from prysm.extras import plot_fourier_chain
from prysm.detector import Detector, OLPF, PixelAperture
from prysm.pupil import Pupil
from prysm.fringezernike import FringeZernike
from prysm.standardzernike import StandardZernike
from prysm.seidel import Seidel
from prysm.psf import PSF, MultispectralPSF, RGBPSF, AiryDisk
from prysm.otf import MTF
from prysm.geometry import (
    gaussian,
    rotated_ellipse,
    square,
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
    Image,
    Slit,
    Pinhole,
    SiemensStar,
    TiltedSquare,
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
    'Seidel',
    'PSF',
    'MultispectralPSF',
    'RGBPSF',
    'AiryDisk',
    'MTF',
    'Lens',
    'gaussian',
    'rotated_ellipse',
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
    'Image',
    'Slit',
    'Pinhole',
    'SiemensStar',
    'TiltedSquare',
]
