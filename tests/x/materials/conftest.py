"""Shared fixtures for the x/materials tests.

A tiny refractiveindex.info-shaped database folder (catalog-nk.yml plus per-page
YAML data files) is written once per session so the RII tests exercise the real
from_database parser without touching the network or the optional package.
"""

import textwrap

import pytest

from prysm.x.materials import RefractiveIndexCatalog


_CATALOG = """\
- SHELF: specs
  name: specs
  content:
    - DIVIDER: "schott"
    - BOOK: SCHOTT-optical
      content:
        - PAGE: N-BK7
          data: schott/N-BK7.yml
- SHELF: glass
  content:
    - BOOK: BK7
      content:
        - PAGE: N-BK7
          data: bk7book/N-BK7.yml
- SHELF: main
  content:
    - BOOK: SiO2
      content:
        - PAGE: Malitson
          data: main/SiO2/Malitson.yml
        - PAGE: Other
          data: main/SiO2/Other.yml
- SHELF: extra
  content:
    - BOOK: HYBRID
      content:
        - PAGE: nk
          data: extra/HYBRID/nk.yml
"""

_FILES = {
    # N-BK7 Sellmeier (ri.info formula 2) -- the canonical SCHOTT spec page.
    'schott/N-BK7.yml': """\
        DATA:
          - type: formula 2
            wavelength_range: 0.3 2.5
            coefficients: 0 1.03961212 0.00600069867 0.231792344 0.0200179144 1.01046945 103.560653
    """,
    # a second, lower-ranked N-BK7 page under a plain glass book.
    'bk7book/N-BK7.yml': """\
        DATA:
          - type: tabulated n
            data: |
              0.4 1.61
              0.6 1.60
              0.8 1.59
    """,
    # tabulated n+k fused silica, two competing references.
    'main/SiO2/Malitson.yml': """\
        DATA:
          - type: tabulated nk
            data: |
              0.4 1.44 0.0
              0.5 1.45 0.001
              0.6 1.46 0.002
    """,
    'main/SiO2/Other.yml': """\
        DATA:
          - type: tabulated nk
            data: |
              0.4 1.55 0.01
              0.6 1.60 0.02
              0.8 1.65 0.03
    """,
    # formula n (same Sellmeier as N-BK7) plus a coarse separate tabulated k:
    # n must stay analytic, not collapse to a 3-point interpolation.
    'extra/HYBRID/nk.yml': """\
        DATA:
          - type: formula 2
            wavelength_range: 0.3 2.5
            coefficients: 0 1.03961212 0.00600069867 0.231792344 0.0200179144 1.01046945 103.560653
          - type: tabulated k
            data: |
              0.3 0.1
              1.0 0.2
              2.5 0.3
    """,
}


@pytest.fixture(scope='session')
def rii_db_path(tmp_path_factory):
    """Write a tiny ri.info-shaped database folder and return its root."""
    root = tmp_path_factory.mktemp('riidb')
    (root / 'catalog-nk.yml').write_text(_CATALOG)
    for rel, body in _FILES.items():
        path = root / 'data' / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(body))
    return root


@pytest.fixture
def rii_catalog(rii_db_path):
    """Build a RefractiveIndexCatalog from the fixture folder (no download)."""
    return RefractiveIndexCatalog.from_database(db_path=rii_db_path, download=False)
