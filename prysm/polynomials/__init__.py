"""Various polynomials of optics."""
# Each family below exposes the canonical surface:
#   <name>, <name>_seq, <name>_der, <name>_der_seq
# Families are listed alphabetically.  Deviations from the canonical surface
# (e.g., a per-axis derivative split, or a Cartesian/polar pair) are noted
# inline.

# Chebyshev (first, second, third, fourth kind)
from .cheby import (  # NOQA
    cheby1,
    cheby1_seq,
    cheby1_der,
    cheby1_der_seq,
    cheby1_2d_sum,
    cheby1_2d_sum_der_xy,
    cheby2,
    cheby2_seq,
    cheby2_der,
    cheby2_der_seq,
    cheby3,
    cheby3_seq,
    cheby3_der,
    cheby3_der_seq,
    cheby4,
    cheby4_seq,
    cheby4_der,
    cheby4_der_seq,
)

# Dickson (first and second kind)
from .dickson import (  # NOQA
    dickson1,
    dickson1_seq,
    dickson1_der,
    dickson1_der_seq,
    dickson2,
    dickson2_seq,
    dickson2_der,
    dickson2_der_seq,
)

# Hermite (probabilist He and physicist H)
from .hermite import (  # NOQA
    hermite_He,
    hermite_He_seq,
    hermite_He_der,
    hermite_He_der_seq,
    hermite_H,
    hermite_H_seq,
    hermite_H_der,
    hermite_H_der_seq,
)

# Jacobi (plus Clenshaw-sum helpers used by Q polynomials and Chebyshev)
from .jacobi import (  # NOQA
    jacobi,
    jacobi_with_der,
    jacobi_seq,
    jacobi_seq_with_der,
    jacobi_der,
    jacobi_der_seq,
    jacobi_sum_clenshaw,
    jacobi_sum_clenshaw_der,
    jacobi_radial_sum,
    jacobi_radial_sum_der_xy,
)

# Laguerre (used for Laguerre-Gaussian beams)
from .laguerre import (  # NOQA
    laguerre,
    laguerre_seq,
    laguerre_der,
    laguerre_der_seq,
)

# Legendre
from .legendre import (  # NOQA
    legendre,
    legendre_seq,
    legendre_der,
    legendre_der_seq,
)

# Forbes Q polynomials.  Q2d_der returns the polar (dr, dt) pair, while
# Q2d_der_xy returns the Cartesian (dx, dy) pair via a harmonic decomposition
# that is finite at the origin.
from .qpoly import (  # NOQA
    Qbfs,
    Qbfs_seq,
    Qbfs_der,
    Qbfs_der_seq,
    Qcon,
    Qcon_seq,
    Qcon_der,
    Qcon_der_seq,
    Q2d,
    Q2d_seq,
    Q2d_der,
    Q2d_der_seq,
    Q2d_der_xy,
    Q2d_der_xy_seq,
)

# XY monomials.  Deliberately exposes three named partial derivatives
# (xy_der_x, xy_der_y, xy_der_xy) plus their _seq variants instead of a single
# xy_der with an axis= kwarg, so the chain rule reads naturally at call sites.
from .xy import (  # NOQA
    xy_j_to_mn,
    xy,
    xy_seq,
    xy_der_x,
    xy_der_x_seq,
    xy_der_y,
    xy_der_y_seq,
    xy_der_xy,
    xy_der_xy_seq,
    xy_sum,
    xy_sum_der_xy,
)

# Zernike (n, m).  zernike_nm_der is the polar (dr, dt) pair; zernike_nm_der_xy
# is the Cartesian (dx, dy) pair.  zernike_sum_der_xy is a Clenshaw-fused
# evaluate-and-differentiate-in-one-pass routine.
from .zernike import (  # NOQA
    zernike_norm,
    zernike_nm,
    zernike_nm_seq,
    zernike_nm_der,
    zernike_nm_der_seq,
    zernike_nm_der_xy,
    zernike_nm_der_xy_seq,
    zernike_sum,
    zernike_sum_der_xy,
    zernikes_to_magnitude_angle,
    zernikes_to_magnitude_angle_nmkey,
    zero_separation as zernike_zero_separation,
    ansi_j_to_nm,
    nm_to_ansi_j,
    nm_to_fringe,
    nm_to_name,
    noll_to_nm,
    fringe_to_nm,
    barplot as zernike_barplot,
    barplot_magnitudes as zernike_barplot_magnitudes,
    top_n,
)

# Mode summation and fitting
from .fitting import (  # NOQA
    sum_of_2d_modes,
    sum_of_2d_modes_adjoint,
    hopkins,
    lstsq,
    normalize_modes,
    orthogonalize_modes,
)
