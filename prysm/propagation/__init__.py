"""Numerical optical propagation.

Files:
- fft               FFT-based pupil <-> focal propagation
- dft               matrix-DFT / chirp-Z propagation with arbitrary sampling
- angular_spectrum  plane-to-plane free space propagation, Fresnel/Talbot metrics
- coronagraph       focal-plane-mask and Babinet/Lyot propagation
- wavefront         the Wavefront type, object oriented interface

fft, dft, angular_spectrum, coronagraph operate on arrays.  Wavefront chains
through the wavefront object for a more "fluent" API and less bookkeeping
"""
from .fft import (
    focus,
    focus_adjoint,
    unfocus,
    unfocus_adjoint,
    Q_for_sampling,
    pupil_sample_to_psf_sample,
    psf_sample_to_pupil_sample,
)
from .dft import (
    coordinates_for_focus,
    prepare_executor,
    prepare_multiresolution,
    MultiResolutionExecutor,
    focus_dft,
    focus_dft_adjoint,
    unfocus_dft,
    unfocus_dft_adjoint,
)
from .angular_spectrum import (
    angular_spectrum,
    angular_spectrum_adjoint,
    angular_spectrum_transfer_function,
    fresnel_number,
    talbot_distance,
)
from .coronagraph import (
    to_fpm_and_back,
    to_fpm_and_back_adjoint,
    to_fpm_and_back_multiresolution,
    to_fpm_and_back_multiresolution_adjoint,
    vortex_phase_mask,
    prepare_measured_fpm,
    babinet,
    babinet_adjoint,
)
from .wavefront import Wavefront

__all__ = [
    'focus',
    'focus_adjoint',
    'unfocus',
    'unfocus_adjoint',
    'Q_for_sampling',
    'pupil_sample_to_psf_sample',
    'psf_sample_to_pupil_sample',
    'coordinates_for_focus',
    'prepare_executor',
    'prepare_multiresolution',
    'MultiResolutionExecutor',
    'focus_dft',
    'focus_dft_adjoint',
    'unfocus_dft',
    'unfocus_dft_adjoint',
    'angular_spectrum',
    'angular_spectrum_adjoint',
    'angular_spectrum_transfer_function',
    'fresnel_number',
    'talbot_distance',
    'to_fpm_and_back',
    'to_fpm_and_back_adjoint',
    'to_fpm_and_back_multiresolution',
    'to_fpm_and_back_multiresolution_adjoint',
    'vortex_phase_mask',
    'prepare_measured_fpm',
    'babinet',
    'babinet_adjoint',
    'Wavefront',
]
