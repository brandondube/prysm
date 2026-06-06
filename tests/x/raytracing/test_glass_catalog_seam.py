"""Contract for the raytracing <-> material-catalog seam.

The raytracing glass layer resolves any object that provides
material_for_name(name) -> callable material.  These tests pin that bridge
against the x/materials AGFCatalog adapter (the prototype backends it replaced
are gone).
"""

from pathlib import Path

import pytest

from prysm.x.materials.agf import AGFCatalog
from prysm.x import materials
from prysm.x.raytracing.io import read_seq, read_zmx


DATA = Path(__file__).parent / 'data' / 'materials'

ZMX_NBK7 = """\
VERS 100000 0
MODE SEQ
UNIT MM
WAVL 0.5875618
SURF 0
  TYPE STANDARD
  DISZ INFINITY
SURF 1
  TYPE STANDARD
  CURV 0.02
  DISZ 5.0
  GLAS N-BK7
SURF 2
  TYPE STANDARD
  CURV 0.0
  DISZ 0.0
"""

SEQ_NBK7 = """\
LEN
CUM
DIM M
WL 587.5618
SO 0 1E10
S 0.02 5 N-BK7_SCHOTT
SI
GO
"""

ZMX_MIRROR_AIR = """\
VERS 100000 0
MODE SEQ
UNIT MM
SURF 0
  TYPE STANDARD
  DISZ INFINITY
SURF 1
  TYPE STANDARD
  CURV 0.0
  DISZ 5.0
  GLAS MIRROR
SURF 2
  TYPE STANDARD
  CURV 0.0
  DISZ 0.0
"""

SEQ_MIRROR_AIR = """\
LEN
CUM
DIM M
SO 0 1E10
S 0 5 REFL
SI
GO
"""


def agf_catalog():
    return AGFCatalog.from_files([
        DATA / 'tiny_schott.agf',
        DATA / 'tiny_ohara.agf',
    ])


def test_lookup_accepts_catalog_adapter():
    material = materials.lookup('N-BK7', database=agf_catalog())
    assert material.name == 'N-BK7'
    assert material(0.5875618) == pytest.approx(1.5168000345)


def test_readers_accept_catalog_adapter():
    catalog = agf_catalog()
    zmx = read_zmx(ZMX_NBK7, _is_text=True, database=catalog)
    seq = read_seq(SEQ_NBK7, _is_text=True, database=catalog)
    assert zmx.surfaces[0].n(0.5875618) == pytest.approx(1.5168000345)
    assert seq.surfaces[0].n(0.5875618) == pytest.approx(1.5168000345)


class ExplodingCatalog:
    def material_for_name(self, name):
        raise AssertionError(f'unexpected material lookup for {name}')


def test_air_and_mirror_reader_paths_bypass_catalog():
    read_zmx(ZMX_MIRROR_AIR, _is_text=True, database=ExplodingCatalog())
    read_seq(SEQ_MIRROR_AIR, _is_text=True, database=ExplodingCatalog())
