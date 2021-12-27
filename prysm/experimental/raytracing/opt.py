"""Minor optimization routines."""

from prysm.conf import config
from prysm.mathops import np

from . import raygen, spencer_and_murty

from scipy import optimize


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
    consider : str, {'x', 'y', 'xy'}
        which ray directions to consider in performing the solve, defaults to
        both X and Y.

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
        # now solve for intersection between the X rays,

        # P for the each ray
        P = phist[-1]
        Px1 = P[0]
        Px2 = P[1]
        Py1 = P[2]
        Py2 = P[3]

        # S for each ray
        S = shist[-1]
        Sx1 = S[0]
        Sx2 = S[1]
        Sy1 = S[2]
        Sy2 = S[3]

        # now use a least squares solution to find the "s" length along the ray directions
        # Ax = y
        # Ax and Ay are for the X and Y fans, not "Ax" which is A@x
        Ax = np.stack([Sx1, -Sx2], axis=1)
        yx = Px2 - Px1
        sx = np.linalg.pinv(Ax) @ yx

        Ay = np.stack([Sy1, -Sy2], axis=1)
        yy = Py2 - Py1
        sy = np.linalg.pinv(Ay) @ yy
        s = np.array([*sx, *sy])
        # fast-forward all the rays and take the average position
        P_out = P + s[:, np.newaxis] * S
        return P_out.mean(axis=0)
        return P


def ray_aim(P, S, prescription, j, wvl, target=(0, 0, np.nan), debug=False):
    """Aim a ray such that it encounters the jth surface at target.

    Parameters
    ----------
    P : numpy.ndarray
        shape (3,), a single ray's initial positions
    S : numpy.ndarray
        shape (3,) a single ray's initial direction cosines
    prescription : iterable
        sequence of surfaces in the prescription
    j : int
        the surface index in prescription at which the ray should hit (target)
    wvl : float
        wavelength of light to use in ray aiming, microns
    target : iterable of length 3
        the position at which the ray should intersect the target surface
        NaNs indicate to ignore that position in aiming
    debug : bool, optional
        if True, returns the (ray-aiming) optimization result as well as the
        adjustment P

    Returns
    -------
    numpy.ndarray
        deltas to P which result in ray intersection

    """
    P = np.asarray(P).astype(config.precision).copy()
    S = np.asarray(S).astype(config.precision).copy()
    target = np.asarray(target)
    trace_path = prescription[:j+1]

    def optfcn(x):
        P[:2] = x
        phist, _ = spencer_and_murty.raytrace(trace_path, P, S, wvl)
        final_position = phist[-1]
        euclidean_dist = (final_position - target)**2
        euclidean_dist = np.nansum(euclidean_dist)/3  # /3 = div by number of axes
        return euclidean_dist

    res = optimize.minimize(optfcn, np.zeros(2), method='L-BFGS-B')
    P[:] = 0
    P[:2] = res.x
    if debug:
        return P, res
    else:
        return P
