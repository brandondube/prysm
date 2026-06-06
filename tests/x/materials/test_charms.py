import pytest

from prysm.x.materials import CHARMSCoefficientMaterial, CHARMSTableMaterial


def test_charms_temperature_sellmeier_constant_coefficients():
    coeffs = {
        'S': [
            [0.6],
            [0.2],
            [0.1],
        ],
        'lambda': [
            [0.1],
            [0.2],
            [10.0],
        ],
    }
    material = CHARMSCoefficientMaterial(
        'test',
        coefficients=coeffs,
        wavelength_range=(0.5, 2.0),
        temperature_range=(20, 300),
    )
    w = 1.0
    n2 = (
        1
        + 0.6 * w ** 2 / (w ** 2 - 0.1 ** 2)
        + 0.2 * w ** 2 / (w ** 2 - 0.2 ** 2)
        + 0.1 * w ** 2 / (w ** 2 - 10.0 ** 2)
    )
    assert material.n(1.0, temperature=77) == pytest.approx(n2 ** 0.5)
    assert material.k(1.0, temperature=77) == pytest.approx(0)


def test_charms_coefficient_array_query_matches_elementwise():
    from prysm.mathops import np

    material = CHARMSCoefficientMaterial(
        'test',
        coefficients={'S': [[0.6], [0.2], [0.1]], 'lambda': [[0.1], [0.2], [10.0]]},
        wavelength_range=(0.5, 2.0),
        temperature_range=(20, 300),
    )
    wq = np.array([0.8, 1.2, 1.6])
    tq = np.array([77.0, 100.0, 150.0])
    out = material.n(wq, temperature=tq)
    assert out.shape == (3,)
    for w, t, value in zip(wq, tq, out):
        assert float(value) == pytest.approx(float(material.n(float(w), temperature=float(t))))


def test_charms_table_material_is_temperature_grid_material():
    material = CHARMSTableMaterial(
        'table',
        [1.0, 2.0],
        [80, 300],
        [
            [1.5, 1.6],
            [1.7, 1.8],
        ],
        layout=('temperature', 'wavelength'),
    )
    assert material.n(1.5, temperature=190) == pytest.approx(1.65)
