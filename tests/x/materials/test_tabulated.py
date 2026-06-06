import pytest

from prysm.mathops import np
from prysm.conf import config
from prysm.x.materials import (
    MaterialRangeError,
    MissingKError,
    TabulatedMaterial,
    TemperatureGridMaterial,
)


def test_tabulated_material_interpolates_n_and_k():
    material = TabulatedMaterial(
        'film',
        [0.4, 0.6, 0.8],
        [1.4, 1.5, 1.6],
        k=[1e-4, 1e-3, 1e-2],
        k_interpolation='log',
    )
    assert material.n(0.5) == pytest.approx(1.45)
    assert material.k(0.5) == pytest.approx(np.sqrt(1e-7))
    nk = material.nk([0.4, 0.8])
    np.testing.assert_allclose(nk, [1.4 + 1e-4j, 1.6 + 1e-2j])


def test_tabulated_material_supports_nearest_and_range_checks():
    material = TabulatedMaterial(
        'nearest',
        [0.4, 0.6, 0.8],
        [1.4, 1.5, 1.6],
        interpolation='nearest',
    )
    assert material.n(0.51) == pytest.approx(1.5)
    with pytest.raises(MaterialRangeError):
        material.n(0.2)


def test_tabulated_material_linear_extrapolates_when_enabled():
    material = TabulatedMaterial(
        'linear',
        [1.0, 2.0],
        [1.0, 3.0],
        extrapolate=True,
    )
    assert material.n(3.0) == pytest.approx(5.0)
    assert material.n(0.0) == pytest.approx(-1.0)


def test_temperature_grid_interpolates_wavelength_and_temperature():
    wavelengths = [0.5, 1.0]
    temperatures = [100, 300]
    # n = 1 + wavelength + temperature / 1000
    n = [
        [1.6, 2.1],
        [1.8, 2.3],
    ]
    material = TemperatureGridMaterial(
        'grid', wavelengths, temperatures, n, layout=('temperature', 'wavelength'),
    )
    assert material.n(0.75, temperature=200) == pytest.approx(1.95)
    np.testing.assert_allclose(
        material.n([0.5, 1.0], temperature=100),
        [1.6, 2.1],
    )
    np.testing.assert_allclose(
        material.k(0.75, temperature=[100, 300]),
        [0, 0],
    )


def test_temperature_grid_extrapolates_wavelength_and_temperature_when_enabled():
    wavelengths = [1.0, 2.0]
    temperatures = [10.0, 20.0]
    # n = wavelength + temperature / 10
    n = [
        [2.0, 3.0],
        [3.0, 4.0],
    ]
    material = TemperatureGridMaterial(
        'grid',
        wavelengths,
        temperatures,
        n,
        extrapolate=True,
        layout=('temperature', 'wavelength'),
    )
    assert material.n(3.0, temperature=30.0) == pytest.approx(6.0)


def test_temperature_grid_uses_derivative_grids_or_finite_difference():
    wavelengths = [0.5, 1.0]
    temperatures = [100, 300]
    n = [
        [1.6, 2.1],
        [1.8, 2.3],
    ]
    material = TemperatureGridMaterial(
        'grid',
        wavelengths,
        temperatures,
        n,
        dn_dT=[
            [1e-3, 1e-3],
            [1e-3, 1e-3],
        ],
        layout=('temperature', 'wavelength'),
    )
    assert material.dn_dT(0.75, 200) == pytest.approx(1e-3)
    assert material.dn_dlambda(0.75, temperature=200) == pytest.approx(1.0, rel=1e-6)


def test_temperature_grid_2d_query_matches_elementwise():
    wavelengths = [0.5, 1.0, 1.5]
    temperatures = [100, 200, 300]
    n = [
        [1.50, 1.55, 1.60],
        [1.52, 1.58, 1.63],
        [1.54, 1.61, 1.66],
    ]
    material = TemperatureGridMaterial(
        'grid', wavelengths, temperatures, n, layout=('temperature', 'wavelength'),
    )
    wq = np.array([[0.6, 0.9], [1.2, 1.4]])
    tq = np.array([[150.0, 250.0], [120.0, 280.0]])
    out = material.n(wq, temperature=tq)
    assert out.shape == (2, 2)
    for i in range(2):
        for j in range(2):
            assert out[i, j] == pytest.approx(
                float(material.n(float(wq[i, j]), temperature=float(tq[i, j])))
            )


def test_tabulated_material_respects_config_precision_and_query_dtype():
    old_precision = config.precision
    try:
        config.precision = np.float32
        material = TabulatedMaterial('film', [0.4, 0.6, 0.8], [1.4, 1.5, 1.6])

        assert material.wavelengths.dtype == np.dtype(np.float32)
        assert material.n([0.5]).dtype == np.dtype(np.float32)
        assert material.n(np.array([0.5], dtype=np.float32)).dtype == np.dtype(np.float32)
        assert material.n(np.array([0.5], dtype=np.float64)).dtype == np.dtype(np.float64)
    finally:
        config.precision = old_precision


def test_temperature_grid_material_respects_config_precision():
    old_precision = config.precision
    try:
        config.precision = np.float32
        material = TemperatureGridMaterial(
            'grid',
            [0.5, 1.0],
            [100, 300],
            [
                [1.6, 2.1],
                [1.8, 2.3],
            ],
            layout=('temperature', 'wavelength'),
        )

        assert material.n_grid.dtype == np.dtype(np.float32)
        assert material.n(0.75, temperature=200).dtype == np.dtype(np.float32)
    finally:
        config.precision = old_precision


def test_temperature_grid_square_layout_disambiguation():
    grid = [[1.6, 2.1], [1.8, 2.3]]
    with pytest.warns(UserWarning, match='square'):
        TemperatureGridMaterial('g', [0.5, 1.0], [100, 300], grid)
    default = TemperatureGridMaterial(
        'g', [0.5, 1.0], [100, 300], grid, layout=('temperature', 'wavelength'),
    )
    transposed = TemperatureGridMaterial(
        'g', [0.5, 1.0], [100, 300], grid, layout=('wavelength', 'temperature'),
    )
    # the two layouts read the off-diagonal entries from opposite axes.
    assert default.n(1.0, temperature=100) == pytest.approx(2.1)
    assert transposed.n(1.0, temperature=100) == pytest.approx(1.8)


def test_temperature_grid_missing_k_raise_is_honored():
    material = TemperatureGridMaterial(
        'g', [0.5, 1.0], [100, 300], [[1.5, 1.6], [1.7, 1.8]],
        missing_k='raise', layout=('temperature', 'wavelength'),
    )
    with pytest.raises(MissingKError):
        material.k(0.75, temperature=200)


def test_temperature_grid_rejects_duplicate_axis_coordinates():
    with pytest.raises(ValueError, match='strictly increasing'):
        TemperatureGridMaterial(
            'g', [0.5, 0.5], [100, 300], [[1.5, 1.6], [1.7, 1.8]],
            layout=('temperature', 'wavelength'),
        )
