"""Merit-function heads: cotangent seeds for the adjoint backward sweep.

Each head exposes seed(trace, intermediates) -> (P_bar, S_bar, L_bar), the
cotangent of the merit with respect to the image-plane ray position, direction,
and (shared across segments) total optical path length.  Full-length N arrays;
invalid rays are zeroed by the backward sweep, but the heads also confine their
seeds to the valid set.

Conventions match prysm.x.raytracing.analysis / opt so the seeds reproduce the
forward-mode tangents:
  - RMS quantities are mean (not sum) of squared deviation over valid rays.
  - OPD is chief-relative (chief == 0), the reference sphere centered on the
    chief image point with radius |P_xp - C|; both move with the chief ray.
"""

from prysm.conf import config
from prysm.mathops import np

from prysm.x.raytracing.opt import _valid_mask, _chief_axis_perp_norm
from prysm.x.raytracing.spencer_and_murty import intersect_reference_sphere
from prysm.x.raytracing.analysis import _pupil_center_chief_index

from .primitives import (
    adj_intersect_reference_sphere_full,
    adj_closest_point_on_axis,
)


def _valid(trace):
    return _valid_mask(trace.status, trace.P[-1])


def _zeros_like_state(trace):
    n = trace.P[-1].shape[0]
    P_bar = np.zeros((n, 3), dtype=config.precision)
    S_bar = np.zeros((n, 3), dtype=config.precision)
    L_bar = np.zeros(n, dtype=config.precision)
    return P_bar, S_bar, L_bar


class RmsSpotSizeSeed:
    """Gradient of the RMS spot size squared (mean radial variance, length^2).

    merit = mean_v |(x_v, y_v) - centroid|^2 over valid rays.  Seeds only the
    transverse landing position; no OPL or direction cotangent.
    """

    name = 'rms_spot_size'

    def seed(self, trace, intermediates):
        P_bar, S_bar, L_bar = _zeros_like_state(trace)
        valid = _valid(trace)
        P_last = trace.P[-1]
        xy = P_last[valid, :2]
        nv = xy.shape[0]
        centroid = xy.mean(axis=0)
        P_bar[valid, 0] = (2.0 / nv) * (xy[:, 0] - centroid[0])
        P_bar[valid, 1] = (2.0 / nv) * (xy[:, 1] - centroid[1])
        return P_bar, S_bar, L_bar

    def value(self, trace, intermediates):
        valid = _valid(trace)
        xy = trace.P[-1][valid, :2]
        centroid = xy.mean(axis=0)
        return float(np.mean(np.sum((xy - centroid) ** 2, axis=1)))


class DistortionSeed:
    """Gradient of the chief ray's image-plane landing coordinate.

    axis in {'x', 'y'} picks the component; merit = P_last[chief, axis].
    Two sweeps (x and y) give the 2 x P Jacobian of chief landing -> distortion.
    """

    def __init__(self, chief_index=None, axis='x'):
        self.chief_index = chief_index
        self.axis = axis
        self.name = f'distortion_{axis}'

    def seed(self, trace, intermediates):
        P_bar, S_bar, L_bar = _zeros_like_state(trace)
        chief = self.chief_index
        if chief is None:
            chief = _pupil_center_chief_index(trace.P[0])
        idx = {'x': 0, 'y': 1}[self.axis]
        P_bar[chief, idx] = 1.0
        return P_bar, S_bar, L_bar

    def value(self, trace, intermediates):
        chief = self.chief_index
        if chief is None:
            chief = _pupil_center_chief_index(trace.P[0])
        return float(trace.P[-1][chief, {'x': 0, 'y': 1}[self.axis]])


class RmsWfeSeed:
    """Gradient of the RMS wavefront error squared on the chief reference sphere.

    merit = mean_v OPD_v^2 (length^2), OPD chief-relative.  Reproduces
    analysis.wavefront / wavefront_with_tangents: the sphere centers on the
    chief image point C and has radius R = |P_xp - C|; both C and (when the
    exit pupil is auto-located) P_xp move with the chief ray, and that coupling
    is seeded back onto the chief ray's position and direction.

    Parameters
    ----------
    chief_index : int, optional
        global index of the chief ray (defaults to the pupil-center ray).
    n_image : float, optional
        image-space index.
    P_xp : iterable, optional
        fixed exit-pupil center; if None the pupil is located as the foot of the
        chief ray's common perpendicular to the optical axis (and its motion is
        differentiated).
    axis_point, axis_dir : iterable, optional
        optical-axis line for the auto exit-pupil (defaults origin, +z).
    """

    def __init__(self, chief_index=None, n_image=1.0, P_xp=None,
                 axis_point=None, axis_dir=None):
        self.chief_index = chief_index
        self.n_image = float(n_image)
        self.P_xp = None if P_xp is None else np.asarray(P_xp, dtype=config.precision)
        self.axis_point = axis_point
        self.axis_dir = axis_dir
        self.name = 'rms_wfe'

    def _geometry(self, trace):
        """Nominal reference-sphere geometry and OPD shared by seed / value."""
        valid = _valid(trace)
        chief = self.chief_index
        if chief is None:
            chief = _pupil_center_chief_index(trace.P[0])
        if not valid[chief]:
            raise ValueError('chief ray is invalid; cannot define reference sphere')

        P_last = trace.P[-1]
        S_last = trace.S[-1]
        C = P_last[chief]
        auto_xp = self.P_xp is None
        if auto_xp:
            axis_point = (np.zeros(3, dtype=config.precision)
                          if self.axis_point is None
                          else np.asarray(self.axis_point, dtype=config.precision))
            axis_dir = (np.array([0., 0., 1.], dtype=config.precision)
                        if self.axis_dir is None
                        else np.asarray(self.axis_dir, dtype=config.precision))
            if _chief_axis_perp_norm(S_last[chief], axis_dir) < 1e-6:
                raise ValueError(
                    'cannot locate the exit pupil from a near-axial chief ray; '
                    'pass P_xp to anchor the reference sphere'
                )
            P_xp = _closest_point_on_axis(C, S_last[chief], axis_point, axis_dir)
        else:
            axis_point = axis_dir = None
            P_xp = self.P_xp
        delta = P_xp - C
        R = float(np.sqrt(np.sum(delta * delta)))
        if R <= 1e-12:
            raise ValueError(
                'reference-sphere radius is degenerate; pass a nondegenerate P_xp'
            )

        _, t = intersect_reference_sphere(P_last[valid], S_last[valid], C, R)
        OPL_total = trace.OPL[:, valid].sum(axis=0) + self.n_image * t
        valid_idx = np.nonzero(valid)[0]
        chief_v = int(np.nonzero(valid_idx == chief)[0][0])
        opd = OPL_total - OPL_total[chief_v]
        return dict(valid=valid, chief=chief, chief_v=chief_v, C=C, R=R,
                    delta=delta, P_xp=P_xp, auto_xp=auto_xp,
                    axis_point=axis_point, axis_dir=axis_dir, opd=opd,
                    P_last=P_last, S_last=S_last)

    def value(self, trace, intermediates):
        return float(np.mean(self._geometry(trace)['opd'] ** 2))

    def seed(self, trace, intermediates):
        P_bar, S_bar, L_bar = _zeros_like_state(trace)
        g = self._geometry(trace)
        valid = g['valid']
        chief = g['chief']
        chief_v = g['chief_v']
        C = g['C']
        R = g['R']
        delta = g['delta']
        auto_xp = g['auto_xp']
        P_last = g['P_last']
        S_last = g['S_last']
        n_image = self.n_image
        opd = g['opd']

        nv = opd.shape[0]
        opd_bar = (2.0 / nv) * opd
        # opl_total cotangent: chief subtraction folds back onto the chief ray
        opl_total_bar = opd_bar.copy()
        opl_total_bar[chief_v] = opl_total_bar[chief_v] - opd_bar.sum()

        # per-ray OPL (segments) and reference-sphere segment cotangents
        L_bar[valid] = opl_total_bar
        t_bar = n_image * opl_total_bar

        P_rs, S_rs, C_bar, R_bar = adj_intersect_reference_sphere_full(
            P_last[valid], S_last[valid], C, R, t_bar)
        P_bar[valid] = P_bar[valid] + P_rs
        S_bar[valid] = S_bar[valid] + S_rs

        # R = |P_xp - C|  ->  delta_bar;  delta = P_xp - C
        if R != 0.0:
            delta_bar = R_bar * delta / R
        else:
            delta_bar = np.zeros(3, dtype=config.precision)
        C_bar = C_bar - delta_bar           # C contributes to delta with -1
        P_xp_bar = delta_bar

        # C is the chief image point
        P_bar[chief] = P_bar[chief] + C_bar
        # auto exit pupil couples back to the chief ray's (P, S)
        if auto_xp:
            P_c_bar, S_c_bar = adj_closest_point_on_axis(
                C, S_last[chief], g['axis_point'], g['axis_dir'], P_xp_bar)
            P_bar[chief] = P_bar[chief] + P_c_bar
            S_bar[chief] = S_bar[chief] + S_c_bar

        return P_bar, S_bar, L_bar


def _closest_point_on_axis(P, S, axis_point, axis_dir):
    """Foot of the common perpendicular from ray (P, S) to the axis line.

    Mirrors opt._closest_approach_on_axis / d_closest_point_on_axis's nominal.
    """
    Sa = axis_dir / np.sqrt(np.sum(axis_dir * axis_dir))
    w = P - axis_point
    a = np.dot(S, S)
    b = np.dot(S, Sa)
    d = np.dot(S, w)
    e = np.dot(Sa, w)
    denom = a * np.dot(Sa, Sa) - b * b
    t = (a * e - b * d) / denom
    return axis_point + t * Sa


__all__ = ['RmsSpotSizeSeed', 'DistortionSeed', 'RmsWfeSeed']
