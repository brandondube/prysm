"""prysm, a python optics module."""
from pkg_resources import get_distribution


from prysm.conf import config
from prysm._richdata import RichData
from prysm.convolution import Convolvable, ConvolutionEngine
from prysm.detector import Detector, OLPF, PixelAperture
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
)
from prysm.objects import (
    Slit,
    Pinhole,
    SiemensStar,
    TiltedSquare,
    SlantedEdge,
    Grating,
    GratingArray,
    Chirp,
)
from prysm.degredations import Smear, Jitter
# from prysm.qpoly import QBFSSag, QCONSag
from prysm.sample_data import sample_files
from prysm.propagation import Wavefront

__all__ = [
    'config',
    'Detector',
    'OLPF',
    'PixelAperture',
    'Interferogram',
    'PSF',
    'AiryDisk',
    'MTF',
    'gaussian',
    'rotated_ellipse',
    'regular_polygon',
    'square',
    'circle',
    'truecircle',
    'Slit',
    'Pinhole',
    'SiemensStar',
    'TiltedSquare',
    'SlantedEdge',
    'Grating',
    'GratingArray',
    'Chirp',
    'Smear',
    'Jitter',
    'Convolvable',
    'ConvolutionEngine',
    'Wavefront',
    'sample_files',
    'RichData',
    'Labels',
    'QBFSSag',
    'QCONSag',
]

__version__ = get_distribution('prysm').version
