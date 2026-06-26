import pytest

from prysm.x.materials import (
    charms_germanium,
    charms_silicon,
    sapphire_ordinary,
    infrared_catalog,
)


def test_charms_silicon_reproduces_measured_index():
    # Frey/Leviton 2006 Table 2: Si n(4.0 um) = 3.42589 @ 295 K, 3.40110 @ 100 K.
    si = charms_silicon()
    assert float(si.n(4.0, temperature=295.0)) == pytest.approx(3.42589, abs=2e-4)
    assert float(si.n(4.0, temperature=100.0)) == pytest.approx(3.40110, abs=2e-4)


def test_charms_germanium_reproduces_measured_index_and_dn_dt():
    # Table 7: Ge n(4.0 um) = 4.02577 @ 295 K, 3.95900 @ 100 K; dn/dT ~ 4e-4 /K.
    ge = charms_germanium()
    assert float(ge.n(4.0, temperature=295.0)) == pytest.approx(4.02577, abs=2e-4)
    assert float(ge.n(4.0, temperature=100.0)) == pytest.approx(3.95900, abs=2e-4)
    dndt = (ge.n(4.0, temperature=296.0) - ge.n(4.0, temperature=294.0)) / 2.0
    assert float(dndt) == pytest.approx(4.0e-4, rel=0.2)


def test_sapphire_ordinary_matches_known_index():
    # Malitson ordinary ray reference: n_o(0.5876 um) ~ 1.7677, n_o(2.0 um) ~ 1.7372.
    sap = sapphire_ordinary()
    assert float(sap.n(0.5876)) == pytest.approx(1.7677, abs=2e-3)
    assert float(sap.n(2.0)) == pytest.approx(1.7372, abs=2e-3)


def test_infrared_catalog_resolves_codev_tokens_at_fixed_temperature():
    ir = infrared_catalog(temperature=295.0)
    ge = ir.material_for_name('GERMMW')
    si = ir.material_for_name('SILICON')
    sap = ir.material_for_name('SAPHIR')
    # the bound CHARMS models answer a bare n(wvl) the ray trace makes
    assert float(ge.n(4.0)) == pytest.approx(4.02577, abs=2e-4)
    assert float(si.n(4.0)) == pytest.approx(3.42589, abs=2e-4)
    assert float(sap.n(2.0)) == pytest.approx(1.7372, abs=2e-3)
