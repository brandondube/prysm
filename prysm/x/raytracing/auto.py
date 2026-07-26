"""Automatic first-order design models."""

from dataclasses import dataclass
import math

from prysm.mathops import np as _np


_NAMES = (
    'efl', 'bfl', 'separation', 'primary_focal_length',
    'primary_to_focus', 'secondary_magnification',
    'primary_radius', 'secondary_radius',
)


@dataclass(frozen=True)
class RCPrescription:
    """Complete Ritchey-Chretien mirror figure prescription."""

    primary_curvature: float
    secondary_curvature: float
    primary_conic: float
    secondary_conic: float


class RitcheyChretien:
    """Partially determined Ritchey-Chretien first-order constraint model.

    Any consistent subset of the supported quantities may be supplied.  The
    model repeatedly closes all algebraically determined quantities, reports
    unresolved values and remaining degrees of freedom, and only emits mirror
    figures or a LensData prescription when the canonical `(efl, bfl,
    separation)` triple is complete.
    """

    def __init__(self, *, efl=None, bfl=None, separation=None,
                 primary_focal_length=None, primary_to_focus=None,
                 secondary_magnification=None, primary_radius=None,
                 secondary_radius=None, rtol=1e-10, atol=1e-12):
        self.rtol = float(rtol)
        self.atol = float(atol)
        self._values = {name: None for name in _NAMES}
        self._sources = {}
        supplied = {
            'efl': efl,
            'bfl': bfl,
            'separation': separation,
            'primary_focal_length': primary_focal_length,
            'primary_to_focus': primary_to_focus,
            'secondary_magnification': secondary_magnification,
            'primary_radius': primary_radius,
            'secondary_radius': secondary_radius,
        }
        self._supplied = {
            name: float(value) for name, value in supplied.items()
            if value is not None
        }
        for name, value in self._supplied.items():
            if not math.isfinite(value):
                raise ValueError(f'{name} must be finite')
            self._set(name, value, f'input {name}')
        self._close()
        self._validate_complete_geometry()

    def _set(self, name, value, source):
        value = float(value)
        current = self._values[name]
        if current is None:
            self._values[name] = value
            self._sources[name] = source
            return True
        if not math.isclose(current, value, rel_tol=self.rtol,
                            abs_tol=self.atol):
            raise ValueError(
                f'inconsistent Ritchey-Chretien constraints for {name}: '
                f'{current:g} from {self._sources[name]} conflicts with '
                f'{value:g} from {source}'
            )
        return False

    def _known(self, *names):
        return all(self._values[name] is not None for name in names)

    def _safe_div(self, numerator, denominator, relation):
        if abs(denominator) <= self.atol:
            raise ValueError(f'singular Ritchey-Chretien constraint: {relation}')
        return numerator / denominator

    def _close(self):
        changed = True
        while changed:
            changed = False
            v = self._values

            if self._known('primary_radius'):
                changed |= self._set(
                    'primary_focal_length', v['primary_radius'] / 2.0,
                    'primary_radius = 2*primary_focal_length')
            if self._known('primary_focal_length'):
                changed |= self._set(
                    'primary_radius', 2.0 * v['primary_focal_length'],
                    'primary_radius = 2*primary_focal_length')

            if self._known('bfl', 'separation'):
                changed |= self._set(
                    'primary_to_focus', v['bfl'] - v['separation'],
                    'primary_to_focus = bfl - separation')
            if self._known('primary_to_focus', 'separation'):
                changed |= self._set(
                    'bfl', v['primary_to_focus'] + v['separation'],
                    'bfl = primary_to_focus + separation')
            if self._known('bfl', 'primary_to_focus'):
                changed |= self._set(
                    'separation', v['bfl'] - v['primary_to_focus'],
                    'separation = bfl - primary_to_focus')
            if self._known('efl', 'primary_to_focus',
                           'secondary_magnification'):
                changed |= self._set(
                    'separation',
                    self._safe_div(
                        v['efl'] - v['primary_to_focus'],
                        v['secondary_magnification'] + 1.0,
                        'secondary magnification is negative one'),
                    'separation = (efl-primary_to_focus)/(magnification+1)')

            if self._known('efl', 'secondary_magnification'):
                changed |= self._set(
                    'primary_focal_length',
                    self._safe_div(-v['efl'], v['secondary_magnification'],
                                   'secondary magnification is zero'),
                    'primary_focal_length = -efl/secondary_magnification')
            if self._known('primary_focal_length',
                           'secondary_magnification'):
                changed |= self._set(
                    'efl', -v['primary_focal_length']
                    * v['secondary_magnification'],
                    'efl = -primary_focal_length*secondary_magnification')
            if self._known('efl', 'primary_focal_length'):
                changed |= self._set(
                    'secondary_magnification',
                    self._safe_div(-v['efl'], v['primary_focal_length'],
                                   'primary focal length is zero'),
                    'secondary_magnification = -efl/primary_focal_length')

            # B, D, and primary focal length close the magnification without
            # needing either F or a secondary constraint:
            # B = F - M D and F = -f1 M -> M = -B/(D+f1).
            if self._known('bfl', 'separation', 'primary_focal_length'):
                changed |= self._set(
                    'secondary_magnification',
                    self._safe_div(
                        -v['bfl'],
                        v['separation'] + v['primary_focal_length'],
                        'separation + primary focal length is zero'),
                    'magnification = -bfl/(separation+primary_focal_length)')

            if self._known('bfl', 'secondary_magnification'):
                changed |= self._set(
                    'secondary_radius',
                    self._safe_div(-2.0 * v['bfl'],
                                   v['secondary_magnification'] - 1.0,
                                   'secondary magnification is one'),
                    'secondary_radius = -2*bfl/(magnification-1)')
            if self._known('secondary_radius', 'secondary_magnification'):
                changed |= self._set(
                    'bfl', -0.5 * v['secondary_radius']
                    * (v['secondary_magnification'] - 1.0),
                    'bfl = -secondary_radius*(magnification-1)/2')
            if self._known('secondary_radius', 'bfl'):
                changed |= self._set(
                    'secondary_magnification',
                    1.0 + self._safe_div(-2.0 * v['bfl'],
                                         v['secondary_radius'],
                                         'secondary radius is zero'),
                    'magnification = 1 - 2*bfl/secondary_radius')

            if self._known('bfl', 'separation',
                           'secondary_magnification'):
                changed |= self._set(
                    'efl', v['bfl'] + v['secondary_magnification']
                    * v['separation'],
                    'efl = bfl + magnification*separation')
            if self._known('efl', 'separation',
                           'secondary_magnification'):
                changed |= self._set(
                    'bfl', v['efl'] - v['secondary_magnification']
                    * v['separation'],
                    'bfl = efl - magnification*separation')
            if self._known('efl', 'bfl', 'secondary_magnification'):
                changed |= self._set(
                    'separation',
                    self._safe_div(v['efl'] - v['bfl'],
                                   v['secondary_magnification'],
                                   'secondary magnification is zero'),
                    'separation = (efl-bfl)/magnification')
            if self._known('efl', 'bfl', 'separation'):
                changed |= self._set(
                    'secondary_magnification',
                    self._safe_div(v['efl'] - v['bfl'], v['separation'],
                                   'separation is zero'),
                    'magnification = (efl-bfl)/separation')

            # The secondary-radius constraint plus any two canonical values
            # closes the third even before B or M is independently known.
            if self._known('efl', 'separation', 'secondary_radius') \
                    and not self._known('bfl'):
                R2, F, D = (v['secondary_radius'], v['efl'], v['separation'])
                M = self._safe_div(R2 - 2.0 * F, R2 - 2.0 * D,
                                   'secondary-radius closure is degenerate')
                changed |= self._set(
                    'secondary_magnification', M,
                    'secondary radius with efl and separation')
            if self._known('efl', 'bfl', 'secondary_radius') \
                    and not self._known('separation'):
                changed |= self._set(
                    'separation',
                    self._safe_div(
                        -v['secondary_radius'] * (v['efl'] - v['bfl']),
                        2.0 * v['bfl'] - v['secondary_radius'],
                        'secondary-radius closure is degenerate'),
                    'secondary radius with efl and bfl')
            if self._known('bfl', 'separation', 'secondary_radius') \
                    and not self._known('efl'):
                changed |= self._set(
                    'efl', v['bfl'] + v['separation']
                    - 2.0 * v['separation'] * v['bfl']
                    / v['secondary_radius'],
                    'secondary radius with bfl and separation')

            # D, f1, and R2 close F directly.  This covers prescriptions
            # specified by the primary and secondary radii plus spacing.
            if self._known('separation', 'primary_focal_length',
                           'secondary_radius') and not self._known('efl'):
                f1, D, R2 = (v['primary_focal_length'], v['separation'],
                             v['secondary_radius'])
                changed |= self._set(
                    'efl',
                    self._safe_div(
                        R2 * f1, 2.0 * (f1 + D) - R2,
                        'mirror-radius closure is degenerate'),
                    'efl from separation and both mirror radii')

    def _validate_complete_geometry(self):
        if not self.complete:
            return
        F, B, D = self.efl, self.bfl, self.separation
        for value, name in ((F, 'efl'), (D, 'separation'),
                            (F - B, 'efl-bfl'),
                            (F - B - D, 'efl-bfl-separation')):
            if abs(value) <= self.atol:
                raise ValueError(
                    f'singular Ritchey-Chretien geometry: {name} is zero')

    @property
    def complete(self):
        return self._known('efl', 'bfl', 'separation')

    @property
    def unresolved(self):
        return tuple(name for name in _NAMES if self._values[name] is None)

    @property
    def degrees_of_freedom(self):
        """Remaining canonical degrees of freedom after supplied constraints."""
        if not self._supplied:
            return 3
        F = self._values['efl'] or 100.0
        B = self._values['bfl'] or 20.0
        D = self._values['separation'] or 30.0
        rows = []
        for name, value in self._supplied.items():
            if name == 'efl':
                rows.append((1.0, 0.0, 0.0))
            elif name == 'bfl':
                rows.append((0.0, 1.0, 0.0))
            elif name == 'separation':
                rows.append((0.0, 0.0, 1.0))
            elif name == 'secondary_magnification':
                rows.append((1.0, -1.0, -value))
            elif name in ('primary_focal_length', 'primary_radius'):
                fp = value if name == 'primary_focal_length' else value / 2.0
                rows.append((D + fp, -fp, F))
            elif name == 'primary_to_focus':
                rows.append((0.0, 1.0, -1.0))
            elif name == 'secondary_radius':
                rows.append((value, -value + 2.0 * D,
                             -value + 2.0 * B))
        rank = int(_np.linalg.matrix_rank(_np.asarray(rows, dtype=float),
                                          tol=self.atol))
        return max(0, 3 - rank)

    @property
    def solutions(self):
        """All discrete complete solutions implied by the supplied inputs.

        A complete model returns itself.  Continuously underdetermined models
        return an empty tuple.  The two three-constraint forms that are
        algebraically closed but have two mirror-layout branches return both,
        rather than silently choosing a branch.
        """
        if self.complete:
            return (self,)
        v = self._values
        candidates = []
        if self._known('efl', 'primary_to_focus', 'secondary_radius'):
            F, p, R2 = (v['efl'], v['primary_to_focus'],
                        v['secondary_radius'])
            # B^2 - (p+R2) B + R2(F+p)/2 = 0.
            roots = _np.roots((1.0, -(p + R2), 0.5 * R2 * (F + p)))
            triples = ((F, float(root.real), float(root.real) - p)
                       for root in roots if abs(float(root.imag)) <= self.atol)
        elif self._known('primary_focal_length', 'primary_to_focus',
                         'secondary_radius'):
            f1, p, R2 = (v['primary_focal_length'], v['primary_to_focus'],
                         v['secondary_radius'])
            # 2D^2 + 2(p+f1-R2)D + 2pf1-R2(p+f1) = 0.
            roots = _np.roots((2.0, 2.0 * (p + f1 - R2),
                               2.0 * p * f1 - R2 * (p + f1)))
            triples = []
            for root in roots:
                if abs(float(root.imag)) > self.atol:
                    continue
                D = float(root.real)
                B = p + D
                if abs(f1 + D) <= self.atol:
                    continue
                F = f1 * B / (f1 + D)
                triples.append((F, B, D))
        else:
            return ()

        for F, B, D in triples:
            try:
                candidate = type(self)(efl=F, bfl=B, separation=D,
                                       rtol=self.rtol, atol=self.atol)
                # Verify every originally supplied constraint on the branch.
                for name, supplied in self._supplied.items():
                    if not math.isclose(getattr(candidate, name), supplied,
                                        rel_tol=self.rtol,
                                        abs_tol=self.atol):
                        break
                else:
                    candidates.append(candidate)
            except ValueError:
                continue
        candidates.sort(key=lambda model: (model.separation, model.bfl,
                                           model.efl))
        return tuple(candidates)

    def prescription(self):
        """Return complete mirror curvatures/conics; reject partial models."""
        if not self.complete:
            discrete = self.solutions
            suffix = (f'; {len(discrete)} discrete solutions are available '
                      'from .solutions' if discrete else '')
            raise ValueError(
                'Ritchey-Chretien model is partially determined; unresolved: '
                + ', '.join(self.unresolved) + suffix
            )
        B, D = self.bfl, self.separation
        M = self.secondary_magnification
        R1 = self.primary_radius
        R2 = self.secondary_radius
        k1 = -1.0 - 2.0 / M ** 3 * B / D
        k2 = -1.0 - 2.0 / (M - 1.0) ** 3 * (
            M * (2.0 * M - 1.0) + B / D)
        return RCPrescription(1.0 / R1, 1.0 / R2, k1, k2)

    def to_lensdata(self, *, primary_aperture=None, secondary_aperture=None):
        """Generate a two-mirror LensData when the model is complete."""
        from .lensdata import LensData
        from .surfaces import Conic

        p = self.prescription()
        lens = LensData()
        lens.add(Conic(p.primary_curvature, p.primary_conic), typ='refl',
                 thickness=self.separation, aperture=primary_aperture)
        lens.add(Conic(p.secondary_curvature, p.secondary_conic), typ='refl',
                 thickness=self.bfl, aperture=secondary_aperture)
        return lens

    def __getattr__(self, name):
        if name in _NAMES:
            return self._values[name]
        raise AttributeError(name)

    def __repr__(self):
        known = ', '.join(
            f'{name}={value:g}' for name, value in self._values.items()
            if value is not None
        )
        return (f'RitcheyChretien({known}; '
                f'degrees_of_freedom={self.degrees_of_freedom})')


__all__ = ['RitcheyChretien', 'RCPrescription']
