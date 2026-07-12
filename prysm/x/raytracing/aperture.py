"""Unified surface aperture: clip, drawn extent, substrate, and edge features."""

from prysm.mathops import np


# ---------------------------------------------------------------------------
# Clip predicates (callables that also carry their radii)
# ---------------------------------------------------------------------------

class CircularClip:
    """Circular clip predicate carrying its radius."""

    def __init__(self, radius, x0=0.0, y0=0.0):
        self.radius = float(radius)
        self.x0 = float(x0)
        self.y0 = float(y0)

    def __call__(self, x, y):
        """True where local coordinates are inside the aperture."""
        dx = x - self.x0
        dy = y - self.y0
        return dx * dx + dy * dy <= self.radius * self.radius

    @property
    def limiting_radius(self):
        """Outer radius that bounds transmitted light."""
        return self.radius

    def __repr__(self):
        return f'CircularClip(radius={self.radius:g})'


class AnnularClip:
    """Annular clip predicate passing the ring, blocking the central disk."""

    def __init__(self, inner_radius, outer_radius, x0=0.0, y0=0.0):
        self.inner_radius = float(inner_radius)
        self.outer_radius = float(outer_radius)
        self.x0 = float(x0)
        self.y0 = float(y0)

    def __call__(self, x, y):
        """True where local coordinates fall in the clear annulus."""
        dx = x - self.x0
        dy = y - self.y0
        rsq = dx * dx + dy * dy
        return (rsq >= self.inner_radius * self.inner_radius) & \
               (rsq <= self.outer_radius * self.outer_radius)

    @property
    def limiting_radius(self):
        """Outer radius that bounds transmitted light."""
        return self.outer_radius

    def __repr__(self):
        return (f'AnnularClip(inner_radius={self.inner_radius:g}, '
                f'outer_radius={self.outer_radius:g})')


def as_aperture(value):
    """Coerce None / float / callable / Aperture into an Aperture."""
    if isinstance(value, Aperture):
        return value
    if value is None:
        return Aperture()
    return Aperture(clip=value)


def circular_aperture(radius, x0=0.0, y0=0.0):
    """Circular clip predicate of the given radius (carries its radius)."""
    return CircularClip(radius, x0, y0)


def annular_aperture(inner_radius, outer_radius, x0=0.0, y0=0.0):
    """Annular clip predicate passing the ring between the radii."""
    return AnnularClip(inner_radius, outer_radius, x0, y0)


# ---------------------------------------------------------------------------
# Drawn extent
# ---------------------------------------------------------------------------

class CircularExtent:
    """Circular (annular when inner_radius > 0) drawn outline.

    Parameters
    ----------
    outer_radius : float
        drawn half-diameter.
    inner_radius : float, optional
        central bore radius; the meridian inside it is masked.

    """

    def __init__(self, outer_radius, inner_radius=0.0):
        self.outer_radius = float(outer_radius)
        self.inner_radius = float(inner_radius)

    def outline(self, points, *, center=0.0, radius=None):
        """Sample a meridian and bore mask; radius overrides outer_radius."""
        r = self.outer_radius if radius is None else radius
        local = np.linspace(-r, r, points)
        ploty = center + local
        mask = np.abs(local) < self.inner_radius
        return ploty, mask

    def __repr__(self):
        if self.inner_radius:
            return (f'CircularExtent(outer_radius={self.outer_radius:g}, '
                    f'inner_radius={self.inner_radius:g})')
        return f'CircularExtent(outer_radius={self.outer_radius:g})'


# ---------------------------------------------------------------------------
# Substrates (mirror backing)
# ---------------------------------------------------------------------------

class Substrate:
    """Cosmetic mirror-substrate drawing model."""

    def __init__(self, thickness, side='auto', bore=0.0):
        self.thickness = float(thickness)
        self.side = side
        self.bore = float(bore)

    def _resolved_side(self, edge_sag):
        """+1 / -1 offset sign; 'auto' infers it from the sag departure."""
        side = self.side
        if isinstance(side, str):
            if side == 'auto':
                departure = np.nanmean(edge_sag - edge_sag[len(edge_sag) // 2])
                return -1.0 if departure > 0 else 1.0
            raise ValueError(f'unknown substrate side {side!r}')
        side = float(side)
        if side == 0:
            raise ValueError('substrate side must be nonzero')
        return np.sign(side)

    def back_sag(self, surf, ploty, sag, edge_sag, center):
        """Rear-face sag along the sampled meridian."""
        raise NotImplementedError

    def back_outline(self, surf, ploty, sag, edge_sag, center, bore=None):
        """Closed meridional outline (zz, tt) of the optical + back faces.

        A bored back renders as two disjoint loops.  bore defaults to the
        substrate bore; a drawn annulus passes its inner radius.
        """
        bore = self.bore if bore is None else float(bore)
        rear_sag = self.back_sag(surf, ploty, sag, edge_sag, center)
        ploty = np.asarray(ploty)
        sag = np.asarray(sag)
        rear_sag = np.asarray(rear_sag)
        if bore > 0.0:
            rear = rear_sag.copy()
            rear[np.abs(ploty - center) < bore] = np.nan
            zz, tt = [], []
            for sel in (ploty >= center + bore,
                        ploty <= center - bore):
                good = sel & np.isfinite(sag) & np.isfinite(rear)
                if not good.any():
                    continue
                fz, rz, py = sag[good], rear[good], ploty[good]
                zz += [*fz, *rz[::-1], fz[0], np.nan]
                tt += [*py, *py[::-1], py[0], np.nan]
            return zz, tt
        zz = [*sag, rear_sag[-1], *rear_sag[::-1], sag[0]]
        tt = [*ploty, ploty[-1], *ploty[::-1], ploty[0]]
        return zz, tt


class SurfaceSubstrate:
    """Optical face only -- no drawn back."""

    bore = 0.0

    def back_outline(self, surf, ploty, sag, edge_sag, center, bore=None):
        """Just the optical face (no back)."""
        return np.asarray(sag), np.asarray(ploty)


class ParallelSubstrate(Substrate):
    """Uniform-thickness shell: back face offset from the optical face."""

    def back_sag(self, surf, ploty, sag, edge_sag, center):
        """Optical sag offset by the (signed) thickness."""
        offset = self._resolved_side(edge_sag) * self.thickness
        return edge_sag + offset


class FlatParentSubstrate(Substrate):
    """Flat back at the parent vertex sag plus thickness."""

    def back_sag(self, surf, ploty, sag, edge_sag, center):
        """Flat plane at the surface vertex z plus the signed thickness."""
        offset = self._resolved_side(edge_sag) * self.thickness
        return np.full_like(np.asarray(edge_sag), surf.P[2] + offset)


class FlatBackSubstrate(Substrate):
    """Flat back parallel to the surface tangent at a reference coordinate."""

    def __init__(self, thickness, side='auto', reference='aperture', bore=0.0):
        super().__init__(thickness, side=side, bore=bore)
        self.reference = reference

    def back_sag(self, surf, ploty, sag, edge_sag, center):
        """Plane through the tangent at the reference transverse coordinate."""
        offset = self._resolved_side(edge_sag) * self.thickness
        ploty = np.asarray(ploty)
        ref = _reference_coordinate(surf, ploty, self.reference)
        # one-sample sag + slope at the reference coordinate
        is_y = True
        coord = np.asarray([ref])
        xpt, ypt = (np.zeros_like(coord), coord)
        z, n_hat = surf.sag_and_normal(xpt, ypt)
        slope = (-n_hat[..., 1] / n_hat[..., 2])[0]
        return surf.P[2] + z[0] + slope * (ploty - ref) + offset


def _reference_coordinate(surf, ploty, reference):
    """Transverse coordinate where a FlatBackSubstrate references its tangent."""
    if not isinstance(reference, str):
        return float(reference)
    reference = reference.lower()
    if reference in ('center', 'centre'):
        return float(np.nanmean(ploty))
    if reference in ('local_vertex', 'section_vertex'):
        return 0.0
    if reference in ('parent', 'parent_vertex'):
        params = surf.params or {}
        return -float(params.get('dy', 0.0))
    if reference in ('aperture', 'near_aperture', 'edge', 'near_edge'):
        parent = _reference_coordinate(surf, ploty, 'parent')
        return float(np.clip(parent, np.nanmin(ploty), np.nanmax(ploty)))
    raise ValueError(f'unknown substrate reference {reference!r}')


# ---------------------------------------------------------------------------
# Edge features (rim-wall cosmetics)
# ---------------------------------------------------------------------------

class EdgeFeature:
    """Base rim-wall cosmetic."""

    is_chamfer = False

    def __init__(self, side='both'):
        self.side = side

    def applies_to(self, wall_side):
        """True when this feature is cut on the given wall ('upper'/'lower')."""
        return self.side in ('both', wall_side)

    def span(self, x0, x1, endpoint_names):
        """(start, end, depth) axial extent of the inset along the wall."""
        raise NotImplementedError


class SquareCut(EdgeFeature):
    """Square step inset between two axial positions."""

    def __init__(self, z_start, z_end, depth, side='both'):
        super().__init__(side)
        self.z_start = z_start
        self.z_end = z_end
        self.depth = depth

    def span(self, x0, x1, endpoint_names):
        """Axial inset extent (z_start, z_end, depth)."""
        return self.z_start, self.z_end, self.depth


class Flat(SquareCut):
    """Flat inset between two axial positions (square-cut geometry)."""


class Chamfer(SquareCut):
    """Diagonal inset between two axial positions."""

    is_chamfer = True


class Seat(EdgeFeature):
    """Stepped seat a fixed width in from a named wall endpoint."""

    def __init__(self, face, width, depth, side='both'):
        super().__init__(side)
        self.face = face
        self.width = width
        self.depth = depth

    def span(self, x0, x1, endpoint_names):
        """Axial inset extent stepping width in from the named face."""
        face = self.face.lower()
        sign = np.sign(x1 - x0)
        if face == endpoint_names[0]:
            return x0, x0 + sign * self.width, self.depth
        if face == endpoint_names[1]:
            return x1 - sign * self.width, x1, self.depth
        raise ValueError('seat face must name one wall endpoint')


# ---------------------------------------------------------------------------
# The aperture
# ---------------------------------------------------------------------------

class Aperture:
    """A surface's clip, drawn extent, oversize, substrate, and rim features.

    Parameters
    ----------
    clip : None, float, or callable
        ray clip.  A float makes a circular clip; None does not clip.
    extent : CircularExtent, optional
        drawn outline, never a clip.  None derives or solves from footprint.
    oversize : float, optional
        ratio from limiting circle to derived drawn extent.
    substrate : Substrate, optional
        mirror backing.
    features : iterable of EdgeFeature, optional
        rim-wall cosmetics.

    """

    def __init__(self, clip=None, *, extent=None, oversize=1.05,
                 substrate=None, features=()):
        if isinstance(clip, (int, float)) and not isinstance(clip, bool):
            clip = circular_aperture(clip)
        self.clip = clip
        self.oversize = float(oversize)
        self.substrate = substrate
        self.features = tuple(features)
        self._user_extent = extent is not None
        self.extent = extent
        # LensData version used by sys.solve.apertures() for auto extents.
        self._solved_at_version = None

    @property
    def is_auto(self):
        """True when no clip and no user-set extent (the solve sizes it)."""
        return self.clip is None and not self._user_extent

    def clips(self, x, y):
        """Boolean mask of rays passing the clip (a scalar True for no clip)."""
        if self.clip is None:
            return np.bool_(True)
        return np.asarray(self.clip(x, y), dtype=bool)

    def limiting_radius(self, footprint=None):
        """Clip radius if the clip exposes one, else the footprint."""
        clip = self.clip
        if clip is not None:
            r = getattr(clip, 'limiting_radius', None)
            if r is not None:
                return r
        return footprint

    def center(self):
        """Local xy center exposed by the clip, else the surface origin."""
        clip = self.clip
        return (float(getattr(clip, 'x0', 0.0)),
                float(getattr(clip, 'y0', 0.0)))

    def drawn_radius(self, footprint=None):
        """Drawn radius: explicit extent, else limiting_radius x oversize."""
        if self.extent is not None:
            return self.extent.outer_radius
        lr = self.limiting_radius(footprint)
        return None if lr is None else lr * self.oversize

    def solve_extent(self, footprint, version, oversize=None):
        """Write a derived circular extent from a traced footprint (the solve)."""
        ov = self.oversize if oversize is None else float(oversize)
        self.extent = CircularExtent(footprint * ov)
        self._user_extent = False
        self._solved_at_version = version

    def is_stale(self, version):
        """True when an auto extent has not been solved against version."""
        if not self.is_auto:
            return False
        return self._solved_at_version != version

    def copy(self):
        """A shallow copy; the extent solve-stamp travels with it."""
        new = Aperture(self.clip, extent=self.extent, oversize=self.oversize,
                       substrate=self.substrate, features=self.features)
        new._user_extent = self._user_extent
        new._solved_at_version = self._solved_at_version
        return new

    def __repr__(self):
        bits = []
        if self.clip is not None:
            bits.append(f'clip={self.clip!r}')
        if self.extent is not None:
            tag = '' if self._user_extent else ' (auto)'
            bits.append(f'extent={self.extent!r}{tag}')
        if self.substrate is not None:
            bits.append(f'substrate={self.substrate!r}')
        if self.features:
            bits.append(f'features={self.features!r}')
        return f'Aperture({", ".join(bits)})'


__all__ = [
    'Aperture',
    'as_aperture',
    'CircularExtent',
    'CircularClip',
    'AnnularClip',
    'circular_aperture',
    'annular_aperture',
    'Substrate',
    'SurfaceSubstrate',
    'ParallelSubstrate',
    'FlatParentSubstrate',
    'FlatBackSubstrate',
    'EdgeFeature',
    'SquareCut',
    'Flat',
    'Chamfer',
    'Seat',
]
