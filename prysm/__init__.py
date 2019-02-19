"""prysm, a python optics module."""
from pkg_resources import get_distribution


from prysm.conf import config
from prysm.convolution import Convolvable
from prysm.detector import Detector, OLPF, PixelAperture
from prysm.pupil import Pupil
from prysm.psf import PSF, AiryDisk
from prysm.otf import MTF
from prysm.interferogram import Interferogram
from prysm.geometry import (
    circle,
    truecircle,
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
from prysm.zernike import FringeZernike, NollZernike, zernikefit
from prysm.sample_data import sample_files

__all__ = [
    'config',
    'Detector',
    'OLPF',
    'PixelAperture',
    'Pupil',
    'FringeZernike',
    'NollZernike',
    'zernikefit',
    'Interferogram',
    'PSF',
    'AiryDisk',
    'MTF',
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
    'circle',
    'truecircle',
    'sample_files',
]

__version__ = get_distribution('prysm').version
