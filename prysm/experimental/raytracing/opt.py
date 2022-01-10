"""Minor optimization routines."""

from prysm.conf import config
from prysm.mathops import np

from . import raygen, spencer_and_murty

from scipy import optimize


def _intersect_lines(P1, S1, P2, S2):
    """Find the slerp along the line (P1, S1) that results in intersection with
    the line (P2, S2).

    P = position, array shape (3,)
    S = direction cosines, array shape (3,)

    pair of two lines only.
    """
    # solution via linear algebra
    Ax = np.stack([S1, -S2], axis=1)
    y = P2 - P1
    return np.linalg.pinv(Ax) @ y


def _establish_axis(P1, P2):
    """Given two points, establish an axis between them.

    Parameters
    ----------
    P1 : numpy.ndarray
        shape (3,), any float dtype
        first point
    P2 : numpy.ndarray
        shape (3,), any float dtype
        second point

    Returns
    -------
    numpy.ndarray, numpy.ndarray
        P1 (same exact PyObject) and direction cosine from P1 -> P2

    """
    diff = P2 - P1
    euclidean_distance = np.sqrt(diff ** 2).sum()
    num = diff
    den = euclidean_distance
    return num / den


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

        # find the distance along line 1 which results in intersection with line 2
        sx = _intersect_lines(Px1, Sx1, Px2, Sx2)
        sy = _intersect_lines(Py1, Sy1, Py2, Sy2)
        s = np.array([*sx, *sy])
        # fast-forward all the rays and take the average position
        P_out = P + s[:, np.newaxis] * S
        return P_out.mean(axis=0)


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


def locate_ep(P_chief, S_chief, P_obj, P_s1):
    """Locate the entrance pupil of a system.

    Note, for a co-axial system P_obj[0] and [1] should be 0, and the same
    is true for P_s1[0] and [1].

    This function,
    1) establishes the axis between the object and the first surface of the system
    2) finds the intersection of the chief ray and that axis

    Parameters
    ----------
    P_chief : numpy.ndarray
        starting position of the chief ray, at the object plane
    S_chief : numpy.ndarray
        starting direction cosine of the chief ray
    P_obj : iterable
        the position of the object

    P_s1 : iterable
        the position of the first surface of the prescription.
        Not the point of intersection for the chief ray, pres[0].P


    Returns
    -------
    numpy.ndarray
        position of the entrance pupil (X,Y,Z)

    """
    S_axis = _establish_axis(P_obj, P_s1)
    s = _intersect_lines(P_chief, S_chief, P_s1, S_axis)
    # s is the slerp for each ray, we just want to go from S1
    return P_s1 + s[1] * S_axis


def locate_xp(P_chief, S_chief, P_img, P_sk):
    """Locate the exit pupil of a system.

    Note, for a co-axial system P_img[0] and [1] should be 0, and the same
    is true for P_sk[0] and [1].

    This function,
    1) establishes the axis between the object and the first surface of the system
    2) finds the intersection of the chief ray and that axis

    Parameters
    ----------
    P_chief : numpy.ndarray
        final position of the chief ray, at the image plane
    S_chief : numpy.ndarray
        final direction cosine of the chief ray
    P_img : iterable
        the position of the object

    P_sk : iterable
        the position of the first surface of the prescription.
        Not the point of intersection for the chief ray, pres[0].P


    Returns
    -------
    numpy.ndarray
        position of the entrance pupil (X,Y,Z)

    """
    S_axis = _establish_axis(P_img, P_sk)
    s = _intersect_lines(P_chief, S_chief, P_sk, S_axis)
    # s is the slerp for each ray, we just want to go from S1
    return P_sk + s[1] * S_axis
