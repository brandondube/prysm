"""Minor optimization routines."""

from prysm.mathops import np

from . import raygen, spencer_and_murty


def paraxial_image_solve(prescription, z, na=0, epd=0, wvl=0.6328):
    """Find the location of the paraxial image.

    The location is found via raytracing and not third-order calculations.

    Two rays are traced very near the optical axis in each X and Y, and the mean
    distance which produces a zero image height is the result of the solve.  If
    na is nonzero, then the ray originates at x=y=0 at 1/1000th of the given NA.

    Parameters
    ----------
    prescription : iterable of Surface
        the prescription to be solved
    z : float
        the z distance (absolute) to solve from
    na : float
        the object-space numerical aperture to use in the solve, if zero the object
        is at infinity, else a finite conjugate.  1/1000th of the given NA is used
        in the solve, the NA of the real system may be quite safely provided
        as an argument.
    epd : float
        entrance pupil diameter, if na=0 and epd=0 an error will be generated.
    wvl : float
        wavelength of light, microns

    Returns
    -------
    numpy.ndarray
        the "P" value to be used with Surface.stop to complete the solve

    """
    if na == 0 and epd == 0:
        raise ValueError("either na or epd must be nonzero")

    PARAXIAL_FRACTION = 1e-4  # 1/1000th
    if na == 0:
        r = epd/2*PARAXIAL_FRACTION
        rayfanx = raygen.generate_collimated_ray_fan(2, maxr=r, azimuth=0)
        rayfany = raygen.generate_collimated_ray_fan(2, maxr=r)
        all_rays = raygen.concat_rayfans(rayfanx, rayfany)
        ps, ss = all_rays
        phist, shist = spencer_and_murty.raytrace(prescription, ps, ss, wvl)
        # now solve for intersection with the optical axis,
        # this code copy-pasted from s&m.py:intersect with modification to
        # solve for x=0/y=0
        P0 = phist[-1]
        S = shist[-1]
        X0 = P0[:2, 0]
        Y0 = P0[2:, 1]
        k = S[:2, 0]  # k, the direction cosine ("x")
        l = S[2:, 1]  # NOQA l direction cosine ("y")
        s0x = -X0/k
        s0y = -Y0/l
        s0xm = s0x.mean()
        s0ym = s0y.mean()
        avg_s0 = sum([s0xm, s0ym])/2
        P = P0[0] + avg_s0 * S[0]  # 0 = the first ray, no need to do an avg raytrace
        return P
