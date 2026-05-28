import numpy as np

from prysm.x.raytracing import materials
from prysm.x.raytracing.analysis import (
    distortion,
    transverse_ray_aberration,
    wavefront,
)
from prysm.x.raytracing.launch import Field, Sampling, launch
from prysm.x.raytracing.opt import rms_spot_radius
from prysm.x.raytracing.paraxial import first_order
from prysm.x.raytracing.spencer_and_murty import raytrace
from prysm.x.raytracing.surfaces import (
    Conic,
    Plane,
    Surface,
    circular_aperture,
)


def _sellmeier(B1, B2, B3, C1, C2, C3):
    def n(w):
        w2 = np.asarray(w) * np.asarray(w)
        return np.sqrt(
            1
            + B1 * w2 / (w2 - C1)
            + B2 * w2 / (w2 - C2)
            + B3 * w2 / (w2 - C3)
        )

    return n


_SK16 = _sellmeier(
    1.343177740,
    0.241144399,
    0.994317969,
    0.00704687339,
    0.0229005004,
    92.7508526,
)
_F2 = _sellmeier(
    1.34533359,
    0.209073176,
    0.937357162,
    0.00997743871,
    0.0470450767,
    111.886764,
)


def _optiland_cooke_triplet():
    rows = [
        (22.01359, 3.25896, _SK16),
        (-435.76044, 6.00755, materials.air),
        (-22.21328, 0.99997, _F2),
        (20.29192, 4.75041, materials.air),
        (79.68360, 2.95208, _SK16),
        (-18.39533, 42.20778, materials.air),
    ]
    z = 0.0
    prescription = []
    for radius, thickness, material in rows:
        prescription.append(
            Surface(
                shape=Conic(1 / radius, 0),
                interaction='refr',
                P=[0, 0, z],
                material=material,
                aperture=circular_aperture(20),
            )
        )
        z += thickness
    prescription.append(
        Surface(shape=Plane(), interaction='eval', P=[0, 0, z],
                material=materials.air)
    )
    return prescription


def test_refracted_direction_cosines_stay_unit_length_on_curved_surfaces():
    presc = _optiland_cooke_triplet()
    P, S = launch(presc, Field(0, 0), 0.55, Sampling.fan(11),
                  epd=10)
    trace = raytrace(presc, P, S, 0.55)
    norms = np.sqrt(np.sum(trace.S * trace.S, axis=-1))
    np.testing.assert_allclose(norms, 1.0, atol=1e-12)


def test_optiland_cooke_first_order_spots_distortion_and_ray_fan():
    presc = _optiland_cooke_triplet()
    fo = first_order(presc[:-1], wvl=0.55, epd=10, stop_index=3)

    np.testing.assert_allclose(fo.efl, 49.999783071431914, atol=3e-4)
    np.testing.assert_allclose(fo.fno, 4.999978307143191, atol=3e-5)
    np.testing.assert_allclose(fo.ep_z, 11.512158673746795, atol=5e-6)
    np.testing.assert_allclose(fo.xp_diameter, 10.233729452318345, atol=1e-6)

    fields = [Field(0, y, unit='deg') for y in (0, 14, 20)]
    wavelengths = [0.48, 0.55, 0.65]
    rms = []
    for field in fields:
        row = []
        for wavelength in wavelengths:
            P, S = launch(presc, field, wavelength, Sampling.hex(nrings=6),
                          epd=10, pupil_z=fo.ep_z)
            trace = raytrace(presc, P, S, wavelength)
            row.append(rms_spot_radius(trace.P[-1], trace.status))
        rms.append(row)
    rms = np.asarray(rms)
    optiland_rms = np.array([
        [0.0037913354614484097, 0.004293689564257648, 0.006195618755672913],
        [0.015824800293446177, 0.016918412809703576, 0.019221165873836536],
        [0.01323623276709289, 0.012116688566407255, 0.013648684944411802],
    ])
    np.testing.assert_allclose(rms, optiland_rms, atol=4e-4)

    _, _, percent = distortion(presc, fields, 0.55, epd=10, pupil_z=fo.ep_z)
    np.testing.assert_allclose(percent[-1], 0.06202477, atol=1e-4)

    P, S = launch(presc, fields[-1], 0.55, Sampling.fan(21),
                  epd=10, pupil_z=fo.ep_z)
    trace = raytrace(presc, P, S, 0.55)
    pupil, fan = transverse_ray_aberration(trace.P, axis='y',
                                           status=trace.status)
    np.testing.assert_allclose(pupil / 5, np.linspace(-1, 1, 21),
                               atol=1e-15)
    optiland_fan = np.array([
        -0.01705141, -0.00449637, 0.00402576, 0.00933527, 0.01209480,
        0.01284530, 0.01203399, 0.01003679, 0.00717719, 0.00374270,
        0.00000000, -0.00379040, -0.00735933, -0.01041300, -0.01261524,
        -0.01356684, -0.01278075, -0.00965171, -0.00341788, 0.00688795,
        0.02250185,
    ])
    np.testing.assert_allclose(fan, optiland_fan, atol=4e-5)

    P_ep, S = launch(presc, fields[-1], 0.55, Sampling.fan(21),
                     epd=10, pupil_z=fo.ep_z)
    z_object_plane = -10.0
    P_obj = P_ep + (((z_object_plane - P_ep[:, 2]) / S[:, 2])[:, np.newaxis] * S)
    opd, x_pup, y_pup = wavefront(
        presc, P_obj, S, 0.55,
        P_xp=(0, 0, fo.xp_z),
        pupil_coords=(P_ep[:, 0], P_ep[:, 1]),
        field=fields[-1],
        output='waves',
    )
    np.testing.assert_allclose(x_pup, P_ep[:, 0], atol=1e-15)
    np.testing.assert_allclose(y_pup / 5, np.linspace(-1, 1, 21),
                               atol=1e-15)
    optiland_opd = np.array([
        0.89768070, 1.05321206, 1.05225981, 0.94883594, 0.78574882,
        0.59683547, 0.40863963, 0.24166768, 0.11130887, 0.02847594,
        0.00000000, 0.02879730, 0.11381106, 0.24971921, 0.42638580,
        0.62801952, 0.83198554, 1.00719504, 1.11196756, 1.09122095,
        0.87278535,
    ])
    np.testing.assert_allclose(opd, optiland_opd, atol=2e-3)
