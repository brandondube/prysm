"""Tests for the Zemax / Code V IO readers and the materials registry."""
import os
import tempfile

import numpy as np
import pytest

from prysm.x import materials
from prysm.x.raytracing.io import read_zmx, read_seq, SurfaceSpec, build_surface
from prysm.x.raytracing.surfaces import (
    Conic, EvenAsphere, Plane, Toroid, Biconic,
    Zernike, XY,
    STYPE_REFLECT, STYPE_REFRACT, STYPE_EVAL,
)
from prysm.x.raytracing.io._indexing import (
    noll_to_nm, fringe_to_nm, xy_j_to_mn,
)
from prysm.x.raytracing.spencer_and_murty import raytrace, valid_mask
from prysm.x.raytracing.launch import Field, Sampling, launch
from prysm.x.raytracing.paraxial import effective_focal_length


def test_surface_spec_builder_constructs_shape_surface():
    surf = build_surface(SurfaceSpec(
        kind='conic',
        typ='refl',
        P=[0.0, 0.0, 0.0],
        params={'c': 1 / 50.0, 'k': -1.0},
    ))
    assert isinstance(surf.shape, Conic)
    assert surf.shape.params is surf.params


# ============================================================================
# materials
# ============================================================================

class FakeMaterial:
    def __init__(self, page, samples):
        self.page = page
        self.samples = np.asarray(samples, dtype=float)

    def get_page_info(self):
        return dict(self.page)

    def get_refractiveindex(self, wvl, unit='nm'):
        assert unit == 'um'
        n = np.interp(wvl, self.samples[:, 0], self.samples[:, 1],
                      left=np.nan, right=np.nan)
        if np.isscalar(wvl):
            return float(n)
        return n


class FakeDatabase:
    def __init__(self, rows):
        self.rows = []
        for pageid, shelf, book, page, samples in rows:
            self.rows.append({
                'pageid': pageid,
                'shelf': shelf,
                'book': book,
                'page': page,
                'filepath': '',
                'hasrefractive': 1,
                'hasextinction': 0,
                'rangeMin': samples[0][0],
                'rangeMax': samples[-1][0],
                'points': len(samples),
                'samples': samples,
            })

    def search_custom(self, sql, params=None):
        key, norm = params
        matches = []
        for row in self.rows:
            page = row['page'].upper()
            page_norm = ''.join(ch for ch in page if ch not in '-_ ')
            if page == key.upper() or page_norm == norm:
                matches.append(tuple(row[k] for k in (
                    'pageid', 'shelf', 'book', 'page', 'filepath',
                    'hasrefractive', 'hasextinction', 'rangeMin',
                    'rangeMax', 'points',
                )))
        return matches

    def get_material(self, pageid):
        for row in self.rows:
            if row['pageid'] == pageid:
                return FakeMaterial(row, row['samples'])
        raise KeyError(pageid)


def _make_database(rows):
    return FakeDatabase(rows)


def test_air_returns_one():
    assert materials.air(0.55) == 1.0


def test_lookup_air_aliases():
    for name in (None, '', 'AIR', 'vacuum', 'Vacuum'):
        f = materials.lookup(name)
        assert f(0.55) == 1.0


def test_lookup_mirror_sentinel():
    sentinel = materials.lookup('MIRROR')
    assert sentinel is materials.MIRROR


def test_lookup_case_insensitive(refractiveindex_database):
    f = materials.lookup('bk7', database=refractiveindex_database)
    np.testing.assert_allclose(float(f(0.587)), 1.5168, atol=1e-3)


def test_lookup_unknown_raises(refractiveindex_database):
    with pytest.raises(KeyError):
        materials.lookup('UNOBTAINIUM', database=refractiveindex_database)


def test_lookup_requires_database_object():
    with pytest.raises(TypeError):
        materials.lookup('BK7', database='/tmp/refractive.db')


@pytest.fixture(autouse=True)
def refractiveindex_database(monkeypatch):
    monkeypatch.setattr(materials, 'Database', FakeDatabase)
    db = _make_database([
        (1, 'specs', 'SCHOTT-optical', 'BK7',
         [(0.4, 1.530849), (0.587, 1.5168), (0.8, 1.510776)]),
        (2, 'specs', 'SCHOTT-optical', 'N-BK7',
         [(0.4, 1.530849), (0.587, 1.5168), (0.8, 1.510776)]),
        (3, 'specs', 'OHARA-optical', 'S-BSL7',
         [(0.4, 1.53), (0.6, 1.52), (0.8, 1.51)]),
    ])
    yield db


def test_refractiveindex_sqlite_glass_lookup_exact_codev_zemax_name():
    db = _make_database([
        (1, 'specs', 'OHARA-optical', 'S-BSL7',
         [(0.4, 1.53), (0.6, 1.52), (0.8, 1.51)]),
    ])
    f = materials.lookup('S-BSL7', database=db)
    np.testing.assert_allclose(float(f(0.5)), 1.525)


def test_refractiveindex_sqlite_prefers_catalog_page_for_n_prefix():
    db = _make_database([
        (1, 'glass', 'BK7', 'N-BK7', [(0.5, 1.61), (0.6, 1.60)]),
        (2, 'specs', 'SCHOTT-optical', 'N-BK7', [(0.5, 1.51), (0.6, 1.50)]),
    ])
    f = materials.lookup('N-BK7', database=db)
    np.testing.assert_allclose(float(f(0.55)), 1.505)
    assert f.page_info['book'] == 'SCHOTT-optical'


def test_refractiveindex_sqlite_out_of_range_returns_nan():
    db = _make_database([
        (1, 'specs', 'OHARA-optical', 'S-BSL7',
         [(0.4, 1.53), (0.6, 1.52), (0.8, 1.51)]),
    ])
    f = materials.lookup('S-BSL7', database=db)
    assert np.isnan(f(0.3))


# ============================================================================
# Zemax .zmx reader
# ============================================================================

_ZMX_SINGLET = """\
VERS 100000 0
MODE SEQ
UNIT MM X W X CM MR CPMM
ENPD 10.0
STOP 1
WAVL 0.55
SURF 0
  TYPE STANDARD
  CURV 0.0
  DISZ INFINITY
SURF 1
  TYPE STANDARD
  CURV 0.02
  DISZ 5.0
  GLAS BK7
SURF 2
  TYPE STANDARD
  CURV -0.02
  DISZ 95.0
SURF 3
  TYPE STANDARD
  CURV 0.0
  DISZ 0.0
"""


def test_zmx_parses_singlet_surface_count(refractiveindex_database):
    pf = read_zmx(_ZMX_SINGLET, _is_text=True, database=refractiveindex_database)
    # OBJECT stripped, S1, S2, IMAGE -> 3 surfaces in prescription
    assert len(pf.surfaces) == 3
    assert pf.source_format == 'zemax'


def test_zmx_singlet_header_fields(refractiveindex_database):
    pf = read_zmx(_ZMX_SINGLET, _is_text=True, database=refractiveindex_database)
    assert pf.epd == 10.0
    assert list(pf.wavelengths.values()) == [0.55]
    assert pf.stop_index == 0  # Zemax SURF 1 -> our index 0
    assert pf.unit == 'mm'


def test_zmx_singlet_curvatures_match_input(refractiveindex_database):
    pf = read_zmx(_ZMX_SINGLET, _is_text=True, database=refractiveindex_database)
    assert isinstance(pf.surfaces[0].shape, Conic)
    assert pf.surfaces[0].params['c'] == 0.02
    assert pf.surfaces[1].params['c'] == -0.02


def test_zmx_singlet_glass_resolves_to_bk7(refractiveindex_database):
    pf = read_zmx(_ZMX_SINGLET, _is_text=True, database=refractiveindex_database)
    np.testing.assert_allclose(float(pf.surfaces[0].n(0.587)), 1.5168, atol=1e-3)


def test_zmx_singlet_image_is_eval_plane(refractiveindex_database):
    pf = read_zmx(_ZMX_SINGLET, _is_text=True, database=refractiveindex_database)
    img = pf.surfaces[-1]
    assert isinstance(img.shape, Plane)
    assert img.typ == STYPE_EVAL


def test_zmx_singlet_vertex_z_stacks_disz(refractiveindex_database):
    pf = read_zmx(_ZMX_SINGLET, _is_text=True, database=refractiveindex_database)
    # vertex_z's:  S1 at 0, S2 at 5, IMAGE at 5+95=100
    assert pf.surfaces[0].P[2] == 0.0
    assert pf.surfaces[1].P[2] == 5.0
    assert pf.surfaces[2].P[2] == 100.0


_ZMX_CLEAR_APERTURE = """\
VERS 100000 0
MODE SEQ
UNIT MM
WAVL 0.55
SURF 0
  TYPE STANDARD
  DISZ INFINITY
SURF 1
  TYPE STANDARD
  CURV 0.0
  DISZ 5.0
  DIAM 1.0
SURF 2
  TYPE STANDARD
  CURV 0.0
  DISZ 0.0
"""


def test_zmx_diam_becomes_clear_aperture_and_clips():
    pf = read_zmx(_ZMX_CLEAR_APERTURE, _is_text=True)
    assert pf.rows[0].semidiameter == 1.0
    P = np.array([[0.0, 0.0, -1.0],
                  [0.0, 1.5, -1.0]])
    S = np.array([[0.0, 0.0, 1.0],
                  [0.0, 0.0, 1.0]])
    tr = raytrace(pf.surfaces, P, S, 0.55)
    assert tr.status.imag[0] == 0
    assert tr.status.imag[1] != 0
    assert tr.status.real[1] == 1


_ZMX_CM_UNITS = """\
VERS 100000 0
MODE SEQ
UNIT CM
ENPD 1.0
WAVL 0.55
SURF 0
  TYPE STANDARD
  DISZ INFINITY
SURF 1
  TYPE STANDARD
  CURV 2.0
  DISZ 0.5
  DIAM 0.2
SURF 2
  TYPE STANDARD
  CURV 0.0
  DISZ 0.0
"""


def test_zmx_non_mm_lengths_scale_to_mm():
    pf = read_zmx(_ZMX_CM_UNITS, _is_text=True)
    assert pf.unit == 'mm'
    assert pf.epd == 10.0
    np.testing.assert_allclose(pf.surfaces[0].params['c'], 0.2)
    np.testing.assert_allclose(pf.surfaces[1].P[2], 5.0)
    np.testing.assert_allclose(pf.rows[0].semidiameter, 2.0)


_ZMX_CM_COORD_BREAK = """\
VERS 100000 0
MODE SEQ
UNIT CM
WAVL 0.55
SURF 0
  TYPE STANDARD
  DISZ INFINITY
SURF 1
  TYPE COORDBRK
  DISZ 0.5
  PARM 1 1.0
  PARM 2 2.0
SURF 2
  TYPE STANDARD
  CURV 0.0
  DISZ 0.0
"""


def test_zmx_non_mm_coordbreak_decenters_scale_to_mm():
    pf = read_zmx(_ZMX_CM_COORD_BREAK, _is_text=True)
    np.testing.assert_allclose(pf.rows[0].decenter, [10.0, 20.0, 0.0])
    np.testing.assert_allclose(pf.rows[0].thickness, 5.0)
    np.testing.assert_allclose(pf.surfaces[0].P, [10.0, 20.0, 5.0])


_ZMX_IMAGE_HEIGHT_FIELD = """\
VERS 100000 0
MODE SEQ
UNIT MM
FTYP 2 0 0 0
XFLN 1.0
YFLN 0.0
SURF 0
  TYPE STANDARD
  DISZ INFINITY
SURF 1
  TYPE STANDARD
  CURV 0.0
  DISZ 0.0
"""


def test_zmx_image_height_fields_are_explicitly_unsupported():
    with pytest.raises(NotImplementedError, match='image-height fields'):
        read_zmx(_ZMX_IMAGE_HEIGHT_FIELD, _is_text=True)


def test_zmx_singlet_round_trips_through_raytrace(refractiveindex_database):
    """Read singlet and run an on-axis collimated trace."""
    pf = read_zmx(_ZMX_SINGLET, _is_text=True, database=refractiveindex_database)
    P, S = launch(pf.surfaces, Field(0., 0.), 0.55,
                  Sampling.fan(n=11), epd=pf.epd, pupil_z=-5.0)
    tr = raytrace(pf.surfaces, P, S, 0.55)
    spot = tr.P[-1, :, :2]
    assert np.isfinite(spot).all()
    assert valid_mask(tr.status, tr.P[-1]).all()
    np.testing.assert_allclose(spot[len(spot) // 2], 0.0, atol=1e-12)
    np.testing.assert_allclose(spot[:, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(spot[:, 1], -spot[::-1, 1], atol=1e-12)


# ---- mirror ----------------------------------------------------------------

_ZMX_PARABOLA = """\
VERS 100000 0
MODE SEQ
UNIT MM
ENPD 10.0
WAVL 0.55
SURF 0
  TYPE STANDARD
  CURV 0.0
  DISZ INFINITY
SURF 1
  TYPE STANDARD
  CURV -0.0125
  CONI -1.0
  DISZ -40.0
  GLAS MIRROR
SURF 2
  TYPE STANDARD
  CURV 0.0
  DISZ 0.0
"""


def test_zmx_mirror_surface_is_reflective():
    pf = read_zmx(_ZMX_PARABOLA, _is_text=True)
    assert pf.surfaces[0].typ == STYPE_REFLECT


def test_zmx_parabola_focuses_to_paraxial_focus():
    pf = read_zmx(_ZMX_PARABOLA, _is_text=True)
    P, S = launch(pf.surfaces, Field(0., 0.), 0.55,
                  Sampling.fan(n=11), epd=10.0, pupil_z=-50.0)
    tr = raytrace(pf.surfaces, P, S, 0.55)
    # parabola is stigmatic on-axis
    spot_y = tr.P[-1, :, 1]
    assert float(np.max(np.abs(spot_y))) < 1e-9


# ---- even asphere ---------------------------------------------------------

_ZMX_EVENASPH = """\
VERS 100000 0
UNIT MM
ENPD 10.0
WAVL 0.55
SURF 0
  TYPE STANDARD
  DISZ INFINITY
SURF 1
  TYPE EVENASPH
  CURV 0.02
  CONI 0.0
  DISZ 5.0
  GLAS BK7
  PARM 1 1.0E-6
  PARM 2 0.0
  PARM 3 -2.0E-12
SURF 2
  TYPE STANDARD
  CURV -0.02
  DISZ 95.0
SURF 3
  TYPE STANDARD
  DISZ 0.0
"""


def test_zmx_evenasph_builds_asphere(refractiveindex_database):
    pf = read_zmx(_ZMX_EVENASPH, _is_text=True, database=refractiveindex_database)
    s1 = pf.surfaces[0]
    assert isinstance(s1.shape, EvenAsphere)
    assert s1.params['coefs'] == (1.0e-6, 0.0, -2.0e-12)


# ---- biconic --------------------------------------------------------------

_ZMX_BICONIC = """\
VERS 100000 0
UNIT MM
ENPD 10.0
WAVL 0.55
SURF 0
  TYPE STANDARD
  DISZ INFINITY
SURF 1
  TYPE BICONICX
  CURV 0.02
  CONI -1.0
  PARM 1 0.015
  PARM 2 -0.5
  DISZ -40.0
  GLAS MIRROR
SURF 2
  TYPE STANDARD
  DISZ 0.0
"""


def test_zmx_biconic_assigns_independent_axes():
    pf = read_zmx(_ZMX_BICONIC, _is_text=True)
    s = pf.surfaces[0]
    assert isinstance(s.shape, Biconic)
    assert s.params['c_x'] == 0.015
    assert s.params['c_y'] == 0.02
    assert s.params['k_x'] == -0.5
    assert s.params['k_y'] == -1.0


# ---- toroid ---------------------------------------------------------------

_ZMX_TOROID = """\
VERS 100000 0
UNIT MM
ENPD 10.0
WAVL 0.55
SURF 0
  TYPE STANDARD
  DISZ INFINITY
SURF 1
  TYPE TOROIDAL
  CURV 0.02
  CONI 0.0
  PARM 1 50.0
  DISZ -40.0
  GLAS MIRROR
SURF 2
  TYPE STANDARD
  DISZ 0.0
"""


def test_zmx_toroid_uses_parm_1_for_rotation_radius():
    pf = read_zmx(_ZMX_TOROID, _is_text=True)
    s = pf.surfaces[0]
    assert isinstance(s.shape, Toroid)
    np.testing.assert_allclose(s.params['c_x'], 1.0 / 50.0)
    np.testing.assert_allclose(s.params['c_y'], 0.02)


# ---- error paths ----------------------------------------------------------

def test_zmx_unsupported_surface_type_raises(refractiveindex_database):
    bad = (_ZMX_SINGLET.replace('TYPE STANDARD\n  CURV 0.02',
                                'TYPE USERSURF\n  CURV 0.02', 1))
    with pytest.raises(NotImplementedError):
        read_zmx(bad, _is_text=True, database=refractiveindex_database)


def test_zmx_empty_file_raises():
    with pytest.raises(ValueError):
        read_zmx('VERS 100000 0\n', _is_text=True)


# ---- file path round-trip -------------------------------------------------

def test_zmx_reads_from_file_path(refractiveindex_database):
    with tempfile.NamedTemporaryFile('w', suffix='.zmx', delete=False) as f:
        f.write(_ZMX_SINGLET)
        path = f.name
    try:
        pf = read_zmx(path, database=refractiveindex_database)
        assert pf.source_path == path
        assert len(pf.surfaces) == 3
    finally:
        os.unlink(path)


# ============================================================================
# Code V .seq reader
# ============================================================================

_SEQ_SINGLET = """\
LEN
RDM
DIM M
WL 587.56
EPD 10.0
SO ; THI 1E10
S ; RDY 50.0 ; THI 5.0 ; GLA BK7
S ; RDY -50.0 ; THI 95.0
SI
GO
"""


def test_seq_singlet_surface_count(refractiveindex_database):
    pf = read_seq(_SEQ_SINGLET, _is_text=True, database=refractiveindex_database)
    # SO is stripped, two real surfaces, SI -> 3 surfaces (two refr + img)
    assert len(pf.surfaces) == 3
    assert pf.source_format == 'codev'


def test_seq_singlet_curvatures_match_radii(refractiveindex_database):
    pf = read_seq(_SEQ_SINGLET, _is_text=True, database=refractiveindex_database)
    np.testing.assert_allclose(pf.surfaces[0].params['c'], 1 / 50.0)
    np.testing.assert_allclose(pf.surfaces[1].params['c'], -1 / 50.0)


def test_seq_singlet_glass_resolves(refractiveindex_database):
    pf = read_seq(_SEQ_SINGLET, _is_text=True, database=refractiveindex_database)
    np.testing.assert_allclose(float(pf.surfaces[0].n(0.587)),
                               1.5168, atol=1e-3)


def test_seq_image_is_eval_plane(refractiveindex_database):
    pf = read_seq(_SEQ_SINGLET, _is_text=True, database=refractiveindex_database)
    img = pf.surfaces[-1]
    assert isinstance(img.shape, Plane)
    assert img.typ == STYPE_EVAL


def test_seq_vertex_z_skips_object_thickness(refractiveindex_database):
    pf = read_seq(_SEQ_SINGLET, _is_text=True, database=refractiveindex_database)
    assert pf.surfaces[0].P[2] == 0.0
    assert pf.surfaces[1].P[2] == 5.0
    assert pf.surfaces[2].P[2] == 100.0


def test_seq_round_trips_through_raytrace(refractiveindex_database):
    pf = read_seq(_SEQ_SINGLET, _is_text=True, database=refractiveindex_database)
    P, S = launch(pf.surfaces, Field(0., 0.), 0.587,
                  Sampling.fan(n=11), epd=pf.epd, pupil_z=-5.0)
    tr = raytrace(pf.surfaces, P, S, 0.587)
    spot = tr.P[-1, :, :2]
    assert float(np.max(np.abs(spot))) < pf.epd / 2.0


_SEQ_CLEAR_APERTURE = """\
LEN
RDM
DIM M
WL 550.0
SO ; THI 1E10
S ; RDY 0.0 ; THI 5.0 ; CAO 1.0
SI
GO
"""


def test_seq_cao_becomes_clear_aperture_and_clips():
    pf = read_seq(_SEQ_CLEAR_APERTURE, _is_text=True)
    assert pf.rows[0].semidiameter == 1.0
    P = np.array([[0.0, 0.0, -1.0],
                  [0.0, 1.5, -1.0]])
    S = np.array([[0.0, 0.0, 1.0],
                  [0.0, 0.0, 1.0]])
    tr = raytrace(pf.surfaces, P, S, 0.55)
    assert tr.status.imag[0] == 0
    assert tr.status.imag[1] != 0
    assert tr.status.real[1] == 1


_SEQ_CM_UNITS = """\
LEN
RDM
DIM CM
WL 550.0
EPD 1.0
SO ; THI 1E10
S ; RDY 5.0 ; THI 0.5 ; CAO 0.2
SI
GO
"""


def test_seq_non_mm_lengths_scale_to_mm():
    pf = read_seq(_SEQ_CM_UNITS, _is_text=True)
    assert pf.unit == 'mm'
    assert pf.epd == 10.0
    np.testing.assert_allclose(pf.surfaces[0].params['c'], 0.02)
    np.testing.assert_allclose(pf.surfaces[1].P[2], 5.0)
    np.testing.assert_allclose(pf.rows[0].semidiameter, 2.0)


_SEQ_IMAGE_HEIGHT_FIELD = """\
LEN
RDM
DIM M
WL 550.0
FNO 7.0
YIM 1.0
XIM 0.0
SO ; THI 1E10
S ; RDY 50.0 ; THI 5.0 ; GLA BK7
S ; RDY -50.0 ; THI 95.0
SI
GO
"""


def test_seq_image_height_fields_become_equivalent_angles(refractiveindex_database):
    pf = read_seq(_SEQ_IMAGE_HEIGHT_FIELD, _is_text=True,
                  database=refractiveindex_database)
    efl = abs(effective_focal_length(pf, wvl=pf.wavelength()))
    expected_hy = np.degrees(np.arctan2(1.0, efl))

    assert len(pf.fields) == 1
    np.testing.assert_allclose(pf.fields[0].hx, 0.0)
    np.testing.assert_allclose(pf.fields[0].hy, expected_hy)
    assert pf.fields[0].kind == 'angle'


def test_seq_fno_becomes_image_space_fnumber(refractiveindex_database):
    pf = read_seq(_SEQ_IMAGE_HEIGHT_FIELD, _is_text=True,
                  database=refractiveindex_database)
    assert pf.aperture.mode == 'FNO_IMAGE'
    assert pf.aperture.value == 7.0
    np.testing.assert_allclose(
        pf.epd,
        abs(effective_focal_length(pf, wvl=pf.wavelength())) / 7.0,
    )


_SEQ_CODEV_VIGNETTING = """\
LEN
RDM
DIM M
WL 550.0
EPD 10.0
YAN 0.0
VUY 0.5
VLY 0.0
SO ; THI 1E10
S ; RDY 0.0 ; THI 5.0
SI
GO
"""


def test_seq_vignetting_factors_clip_launched_pupil():
    pf = read_seq(_SEQ_CODEV_VIGNETTING, _is_text=True)
    assert pf.fields[0].vignetting == {
        'vux': 0.0, 'vlx': 0.0, 'vuy': 0.5, 'vly': 0.0,
    }

    P, _ = launch(pf, pf.field(0), pf.wavelength(), Sampling.fan(n=5, axis='y'))
    assert float(np.max(P[:, 1])) <= 2.5
    assert float(np.min(P[:, 1])) == -5.0


# ---- STO (aperture stop) ---------------------------------------------------

_SEQ_STOP = """\
LEN
RDM
DIM M
WL 550.0
EPD 10.0
SO ; THI 1E10
S ; RDY 50.0 ; THI 5.0
STO
S ; RDY -50.0 ; THI 95.0
SI
GO
"""


def test_seq_sto_marks_the_surface_it_appears_on():
    """STO sets the stop on the surface it follows, not one surface later.

    Regression: the ordinal counted the object surface, which the downstream
    real_counter walk skips, so the stop landed one real surface too late.
    STO here follows the first real surface, so stop_index must be 0.
    """
    ld = read_seq(_SEQ_STOP, _is_text=True)
    assert ld.stop_index == 0


# ---- mirror ----------------------------------------------------------------

_SEQ_PARABOLA = """\
LEN
DIM M
WL 550.0
EPD 10.0
SO ; THI 1E10
S ; CUY -0.0125 ; K -1.0 ; THI -40.0 ; GLA REFL
SI
GO
"""


def test_seq_mirror_keyword_yields_reflective():
    pf = read_seq(_SEQ_PARABOLA, _is_text=True)
    assert pf.surfaces[0].typ == STYPE_REFLECT


def test_seq_parabola_focuses_on_axis():
    pf = read_seq(_SEQ_PARABOLA, _is_text=True)
    P, S = launch(pf.surfaces, Field(0., 0.), 0.55,
                  Sampling.fan(n=11), epd=10.0, pupil_z=-50.0)
    tr = raytrace(pf.surfaces, P, S, 0.55)
    assert float(np.max(np.abs(tr.P[-1, :, 1]))) < 1e-9


# ---- even asphere via A/B/C/D coefs ---------------------------------------

_SEQ_ASPH = """\
LEN
DIM M
WL 550.0
EPD 10.0
SO ; THI 1E10
S ; RDY 50.0 ; THI 5.0 ; GLA BK7
ASP
A 1.0E-6
B 0.0
C -2.0E-12
S ; RDY -50.0 ; THI 95.0
SI
GO
"""


def test_seq_A_through_C_coefs_become_asphere(refractiveindex_database):
    pf = read_seq(_SEQ_ASPH, _is_text=True, database=refractiveindex_database)
    s1 = pf.surfaces[0]
    assert isinstance(s1.shape, EvenAsphere)
    assert s1.params['coefs'] == (1.0e-6, 0.0, -2.0e-12)


# ---- header parsing -------------------------------------------------------

_SEQ_HEADER = """\
LEN
TITLE A small singlet
DIM M
WL 486.13 587.56 656.27
REF 2
EPD 5.0
YAN 0.0 1.0 2.0
SO ; THI 1E10
S ; RDY 30 ; THI 2 ; GLA BK7
S ; RDY -30 ; THI 50
SI
GO
"""


def test_seq_header_wavelengths_and_reference(refractiveindex_database):
    pf = read_seq(_SEQ_HEADER, _is_text=True, database=refractiveindex_database)
    # Code V WL is nanometers; the reader converts to microns.
    np.testing.assert_allclose(list(pf.wavelengths.values()),
                               [0.48613, 0.58756, 0.65627])


def test_seq_yan_becomes_field_list(refractiveindex_database):
    pf = read_seq(_SEQ_HEADER, _is_text=True, database=refractiveindex_database)
    assert len(pf.fields) == 3
    assert pf.fields[0].hy == 0.0
    assert pf.fields[1].hy == 1.0
    assert pf.fields[2].hy == 2.0
    assert all(f.kind == 'angle' for f in pf.fields)


def test_seq_empty_text_raises():
    with pytest.raises(ValueError):
        read_seq('LEN\nGO\n', _is_text=True)


def test_seq_reads_from_file_path(refractiveindex_database):
    with tempfile.NamedTemporaryFile('w', suffix='.seq', delete=False) as f:
        f.write(_SEQ_SINGLET)
        path = f.name
    try:
        pf = read_seq(path, database=refractiveindex_database)
        assert pf.source_path == path
        assert len(pf.surfaces) == 3
    finally:
        os.unlink(path)


# ============================================================================
# Indexing tables
# ============================================================================

def test_noll_to_nm_first_six():
    assert noll_to_nm(1) == (0, 0)
    assert noll_to_nm(2) == (1, 1)
    assert noll_to_nm(3) == (1, -1)
    assert noll_to_nm(4) == (2, 0)
    # j=5/6 are the (2, +/-2) astigmatism pair; sign order varies between
    # references but |m|=2 and n=2 are the invariant
    n5, m5 = noll_to_nm(5)
    n6, m6 = noll_to_nm(6)
    assert n5 == 2 and n6 == 2
    assert {abs(m5), abs(m6)} == {2}


def test_fringe_to_nm_first_nine():
    assert fringe_to_nm(1) == (0, 0)
    assert fringe_to_nm(2) == (1, 1)
    assert fringe_to_nm(3) == (1, -1)
    assert fringe_to_nm(4) == (2, 0)
    n9, m9 = fringe_to_nm(9)
    assert (n9, m9) == (4, 0)  # primary spherical


def test_xy_j_to_mn_first_six_matches_codev_convention():
    assert xy_j_to_mn(1) == (0, 0)
    assert xy_j_to_mn(2) == (1, 0)
    assert xy_j_to_mn(3) == (0, 1)
    assert xy_j_to_mn(4) == (2, 0)
    assert xy_j_to_mn(5) == (1, 1)
    assert xy_j_to_mn(6) == (0, 2)


def test_xypoly_total_degree_ordering():
    """Term j sits at total degree i where 1+2+...+i < j <= 1+2+...+(i+1)."""
    # j=7 should be (3, 0) — start of degree-3 block
    assert xy_j_to_mn(7) == (3, 0)
    # j=10 is the last degree-3 term: (0, 3)
    assert xy_j_to_mn(10) == (0, 3)


# ============================================================================
# Zemax ZERNSAG / XYPOLY
# ============================================================================

_ZMX_ZERNSAG = """\
VERS 100000 0
UNIT MM
ENPD 10.0
WAVL 0.55
SURF 0
  TYPE STANDARD
  DISZ INFINITY
SURF 1
  TYPE ZERNSAG
  CURV -0.0125
  CONI -1.0
  DISZ -40.0
  GLAS MIRROR
  PARM 1 5.0
  XDAT 1 0.0 0 0
  XDAT 2 0.0 0 0
  XDAT 3 0.0 0 0
  XDAT 4 1.0e-4 0 0
  XDAT 5 0.0 0 0
  XDAT 6 0.0 0 0
SURF 2
  TYPE STANDARD
  DISZ 0.0
"""


def test_zmx_zernsag_builds_surface_zernike():
    pf = read_zmx(_ZMX_ZERNSAG, _is_text=True)
    assert isinstance(pf.surfaces[0].shape, Zernike)
    params = pf.surfaces[0].params
    assert params['normalization_radius'] == 5.0
    # PARM 1 → norm radius; 6 XDAT terms set
    assert len(params['nms']) == 6
    assert len(params['coefs']) == 6
    assert params['coefs'][3] == 1.0e-4  # j=4 has coef 1e-4
    assert params['norm'] is True


def test_zmx_zernsag_missing_norm_radius_raises():
    bad = _ZMX_ZERNSAG.replace('PARM 1 5.0', 'PARM 1 0.0')
    with pytest.raises(ValueError):
        read_zmx(bad, _is_text=True, database=refractiveindex_database)


def test_zmx_zernsag_no_coefs_falls_back_to_conic():
    """A ZERNSAG with norm radius but no XDAT degenerates to a Conic."""
    txt = (
        'UNIT MM\nENPD 10.0\nWAVL 0.55\n'
        'SURF 0\n  TYPE STANDARD\n  DISZ INFINITY\n'
        'SURF 1\n  TYPE ZERNSAG\n  CURV -0.0125\n  CONI -1.0\n'
        '  DISZ -40.0\n  GLAS MIRROR\n  PARM 1 5.0\n'
        'SURF 2\n  TYPE STANDARD\n  DISZ 0.0\n'
    )
    pf = read_zmx(txt, _is_text=True)
    assert isinstance(pf.surfaces[0].shape, Conic)


_ZMX_XYPOLY = """\
VERS 100000 0
UNIT MM
ENPD 10.0
WAVL 0.55
SURF 0
  TYPE STANDARD
  DISZ INFINITY
SURF 1
  TYPE XYPOLY
  CURV -0.0125
  CONI -1.0
  DISZ -40.0
  GLAS MIRROR
  PARM 1 10.0
  XDAT 1 0.0 0 0
  XDAT 2 0.0 0 0
  XDAT 3 0.0 0 0
  XDAT 4 2.0e-5 0 0
SURF 2
  TYPE STANDARD
  DISZ 0.0
"""


def test_zmx_xypoly_builds_surface_xy():
    pf = read_zmx(_ZMX_XYPOLY, _is_text=True)
    assert isinstance(pf.surfaces[0].shape, XY)
    params = pf.surfaces[0].params
    assert params['normalization_radius'] == 10.0
    # 4 terms parsed
    assert len(params['mns']) == 4
    # j=4 maps to (m=2, n=0): pure x^2 monomial
    assert params['mns'][3] == (2, 0)
    assert params['coefs'][3] == 2.0e-5


def test_zmx_xypoly_default_norm_radius_when_zero():
    """PARM 1 = 0 (or absent) yields normalization_radius = 1.0."""
    txt = _ZMX_XYPOLY.replace('PARM 1 10.0', 'PARM 1 0.0')
    pf = read_zmx(txt, _is_text=True)
    assert pf.surfaces[0].params['normalization_radius'] == 1.0


# ============================================================================
# Code V biconic
# ============================================================================

_SEQ_BICONIC = """\
LEN
DIM M
WL 550.0
EPD 10.0
SO ; THI 1E10
S ; RDY 100.0 ; RDX 50.0 ; K -1.0 ; KX -0.5 ; THI -40.0 ; GLA REFL
SI
GO
"""


def test_seq_biconic_built_when_x_axis_present():
    pf = read_seq(_SEQ_BICONIC, _is_text=True)
    assert isinstance(pf.surfaces[0].shape, Biconic)
    p = pf.surfaces[0].params
    np.testing.assert_allclose(p['c_y'], 1 / 100.0)
    np.testing.assert_allclose(p['c_x'], 1 / 50.0)
    assert p['k_y'] == -1.0
    assert p['k_x'] == -0.5


def test_seq_no_x_axis_stays_conic(refractiveindex_database):
    """Without RDX/CUX/CCX, S is just a conic, not a biconic."""
    pf = read_seq(_SEQ_SINGLET, _is_text=True, database=refractiveindex_database)
    assert isinstance(pf.surfaces[0].shape, Conic)


# ============================================================================
# Code V ZFR (Fringe Zernike)
# ============================================================================

_SEQ_ZFR = """\
LEN
DIM M
WL 550.0
EPD 10.0
SO ; THI 1E10
S ; CUY -0.0125 ; K -1.0 ; THI -40.0 ; GLA REFL
NRR 5.0
ZFR 0.0 0.0 0.0 1.0e-4 0.0 0.0
SI
GO
"""


def test_seq_zfr_builds_zernike_surface_with_fringe_norm():
    pf = read_seq(_SEQ_ZFR, _is_text=True)
    assert isinstance(pf.surfaces[0].shape, Zernike)
    p = pf.surfaces[0].params
    assert p['normalization_radius'] == 5.0
    assert p['norm'] is False  # Fringe is non-orthonormalized
    assert len(p['coefs']) == 6
    assert p['coefs'][3] == 1.0e-4


def test_seq_zfr_default_norm_radius_is_one():
    """When NRR is omitted, normalization radius defaults to 1.0."""
    txt = _SEQ_ZFR.replace('NRR 5.0\n', '')
    pf = read_seq(txt, _is_text=True)
    assert pf.surfaces[0].params['normalization_radius'] == 1.0


# ============================================================================
# Code V XYP
# ============================================================================

_SEQ_XYP = """\
LEN
DIM M
WL 550.0
EPD 10.0
SO ; THI 1E10
S ; CUY -0.0125 ; K -1.0 ; THI -40.0 ; GLA REFL
NRR 10.0
XYP 0.0 0.0 0.0 2.0e-5
SI
GO
"""


def test_seq_xyp_builds_surface_xy():
    pf = read_seq(_SEQ_XYP, _is_text=True)
    assert isinstance(pf.surfaces[0].shape, XY)
    p = pf.surfaces[0].params
    assert p['normalization_radius'] == 10.0
    # term 4 corresponds to (m=2, n=0)
    assert p['mns'][3] == (2, 0)
    assert p['coefs'][3] == 2.0e-5


# ============================================================================
# Code V XDE / YDE / ADE / BDE / CDE
# ============================================================================

_SEQ_DECENTER = """\
LEN
DIM M
WL 550.0
EPD 4.0
SO ; THI 1E10
S ; RDY 30 ; THI 2.0 ; GLA BK7
DAR
XDE 0.5 ; YDE 0.25
ADE 1.0
BDE -2.0
S ; RDY -30 ; THI 50.0
SI
GO
"""


def test_seq_decentered_surface_has_decenter(refractiveindex_database):
    pf = read_seq(_SEQ_DECENTER, _is_text=True, database=refractiveindex_database)
    s = pf.surfaces[0]
    # XDE/YDE was added to the surface position
    np.testing.assert_allclose(s.P[0], 0.5)
    np.testing.assert_allclose(s.P[1], 0.25)


def test_seq_decentered_surface_has_rotation(refractiveindex_database):
    pf = read_seq(_SEQ_DECENTER, _is_text=True, database=refractiveindex_database)
    s = pf.surfaces[0]
    # ADE=1, BDE=-2 -> a non-identity rotation
    assert s.R is not None
    # identity check: max deviation from eye(3) should be > 1e-4 for the
    # specified non-zero tilt
    R = np.asarray(s.R, dtype=float)
    eye = np.eye(3)
    assert float(np.max(np.abs(R - eye))) > 1e-4


def test_seq_undecentered_surface_has_no_rotation(refractiveindex_database):
    """The second surface in _SEQ_DECENTER was not decentered."""
    pf = read_seq(_SEQ_DECENTER, _is_text=True, database=refractiveindex_database)
    s2 = pf.surfaces[1]
    assert s2.R is None
    np.testing.assert_allclose(s2.P[:2], (0.0, 0.0))


# ============================================================================
# Code V free-format (positional) surface lines -- the form real Code V writes
# ============================================================================

_SEQ_POSITIONAL = """\
RDM
LEN
TITLE 'positional singlet'
DIM M
WL 587.6
EPD 10.0
SO    0. 1E10
S     50.0 5.0 BK7
S     -50.0 95.0
SI    0. 0.
GO
"""


def test_seq_positional_radius_thickness_glass(refractiveindex_database):
    """S <radius> <thickness> <glass> positional free format parses."""
    pf = read_seq(_SEQ_POSITIONAL, _is_text=True,
                  database=refractiveindex_database)
    np.testing.assert_allclose(pf.surfaces[0].params['c'], 1 / 50.0)
    np.testing.assert_allclose(pf.surfaces[1].params['c'], -1 / 50.0)
    np.testing.assert_allclose(float(pf.surfaces[0].n(0.587)), 1.5168,
                               atol=1e-3)
    # SO/S/SI consume their positional thickness: vertices stack the gaps.
    assert pf.surfaces[0].P[2] == 0.0
    assert pf.surfaces[1].P[2] == 5.0
    assert pf.surfaces[2].P[2] == 100.0


def test_seq_wavelength_converted_nm_to_um(refractiveindex_database):
    """Code V WL is nanometers; the reader stores microns."""
    pf = read_seq(_SEQ_POSITIONAL, _is_text=True,
                  database=refractiveindex_database)
    np.testing.assert_allclose(list(pf.wavelengths.values()), [0.5876])


_SEQ_K_CONIC = """\
RDM
LEN
DIM M
WL 550.0
EPD 10.0
SO    0. 1E10
S     -80.0 -40.0 REFL
  CCY 0
  K   -1.0
SI    0. 0.
GO
"""


def test_seq_K_sets_conic_and_CCY_is_a_control_code():
    """K is the conic constant; CCY 0 is a coupling code and is ignored."""
    pf = read_seq(_SEQ_K_CONIC, _is_text=True)
    s = pf.surfaces[0]
    np.testing.assert_allclose(s.params['c'], -1 / 80.0)
    np.testing.assert_allclose(s.params['k'], -1.0)
    assert s.typ == STYPE_REFLECT


def test_seq_stop_without_object_surface():
    """STO resolves even when no SO object surface is written.

    threemir.seq starts straight at S with the stop on the first surface; the
    old ordinal math assumed a committed object and lost the stop.
    """
    txt = (
        'RDM\nLEN\nDIM M\nWL 550.0\nEPD 10.0\n'
        'S 0. 10.0\n  STO\n'
        'S 50 5\nS -50 90\nSI 0 0\nGO\n'
    )
    pf = read_seq(txt, _is_text=True)
    assert pf.stop_index == 0


_SEQ_DAR_VS_BASIC = """\
RDM
LEN
DIM M
WL 550.0
EPD 10.0
SO    0. 1E10
S     -100. -20. REFL
  YDE 10.0
  ADE 5.0
S     -100. -20. REFL
  DAR
  YDE 3.0
  ADE 2.0
S     0. 30.
SI    0. 0.
GO
"""


def test_seq_dar_is_local_basic_persists():
    """A bare decenter is a persistent (basic) break; DAR returns the axis.

    Surface 0 has a basic YDE/ADE, so surface 1 (DAR) and the surface after it
    inherit that frame; the DAR on surface 1 applies to surface 1 only.
    """
    pf = read_seq(_SEQ_DAR_VS_BASIC, _is_text=True)
    s0, s1, s2 = pf.surfaces[0], pf.surfaces[1], pf.surfaces[2]
    # surface 0 (basic) is decentered up 10mm
    np.testing.assert_allclose(float(s0.P[1]), 10.0)
    # surface 1 carries the persistent basic decenter plus its own DAR offset
    assert s1.R is not None
    # surface 2 has no break of its own but inherits the persistent basic
    # frame from surface 0 (tilted), so it is rotated -- not identity.
    assert s2.R is not None


def test_seq_glass_catalog_suffix_stripped(refractiveindex_database):
    """Code V GLA names are GLASS_CATALOG; the bare glass still resolves."""
    txt = (
        'RDM\nLEN\nDIM M\nWL 587.6\nEPD 10.0\n'
        'SO 0. 1E10\nS 50 5 BK7_SCHOTT\nS -50 95\nSI 0 0\nGO\n'
    )
    pf = read_seq(txt, _is_text=True, database=refractiveindex_database)
    np.testing.assert_allclose(float(pf.surfaces[0].n(0.587)), 1.5168,
                               atol=1e-3)
