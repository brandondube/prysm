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
    P1 : ndarray
        shape (3,), any float dtype
        first point
    P2 : ndarray
        shape (3,), any float dtype
        second point

    Returns
    -------
    ndarray, ndarray
        P1 (same exact PyObject) and direction cosine from P1 -> P2

    """
    diff = P2 - P1
    # L2 norm: prior code computed sqrt(diff**2).sum() which is the L1 norm and
    # only accidentally equals L2 when diff has a single nonzero component
    # (e.g. coaxial systems with diff = [0,0,dz]).
    euclidean_distance = np.sqrt(np.sum(diff * diff))
    return diff / euclidean_distance


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
    ndarray
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
        phist, shist, _ = spencer_and_murty.raytrace(prescription, ps, ss, wvl)
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
    P : ndarray
        shape (3,), a single ray's initial positions
    S : ndarray
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
    ndarray
        deltas to P which result in ray intersection

    """
    P = np.asarray(P).astype(config.precision).copy()
    S = np.asarray(S).astype(config.precision).copy()
    target = np.asarray(target)
    trace_path = prescription[:j+1]

    def optfcn(x):
        P[:2] = x
        phist, _, _ = spencer_and_murty.raytrace(trace_path, P, S, wvl)
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


def _closest_approach_on_axis(P_chief, S_chief, axis_point, axis_dir):
    """Point on (axis_point, axis_dir) closest to the chief ray.

    For a centered system, the optical axis is z-hat through the origin and
    the exit (or entrance) pupil center lies where the chief ray crosses it.
    Real chief rays in 3D are usually skew to that axis, so we return the
    point on the axis at the foot of the common perpendicular.

    """
    A = np.asarray(P_chief)
    Sc = np.asarray(S_chief)
    B = np.asarray(axis_point)
    Sa = np.asarray(axis_dir)
    Sa = Sa / np.sqrt(np.sum(Sa * Sa))
    w = A - B
    a = np.sum(Sc * Sc)
    b = np.sum(Sc * Sa)
    c = np.sum(Sa * Sa)
    d = np.sum(Sc * w)
    e = np.sum(Sa * w)
    denom = a * c - b * b
    if abs(denom) < 1e-30:
        # chief ray parallel to axis -> pupil at infinity along the axis
        # return the projection of A onto the axis, with a NaN flag in z if you like
        t = e / c
        return B + t * Sa
    t = (a * e - b * d) / denom
    return B + t * Sa


def locate_ep(P_chief, S_chief, P_obj, P_s1):
    """Locate the entrance pupil of a system.

    Defines the optical axis as the line through P_obj and P_s1, then finds
    the point on that axis closest to the chief ray.  For a coaxial system
    this reduces to the standard z-axis intersection; for tilted/decentered
    systems the user should supply a meaningful axis pair (P_obj, P_s1).

    Parameters
    ----------
    P_chief : ndarray
        any point on the chief ray (e.g. its starting position at the object)
    S_chief : ndarray
        chief ray direction cosines
    P_obj : iterable
        object-side point on the optical axis (commonly the object location)
    P_s1 : iterable
        a second axis point (commonly the first surface vertex).  Must differ
        from P_obj.

    Returns
    -------
    ndarray
        position of the entrance pupil (X,Y,Z)

    """
    S_axis = _establish_axis(np.asarray(P_obj), np.asarray(P_s1))
    return _closest_approach_on_axis(P_chief, S_chief, np.asarray(P_obj), S_axis)


def xp_reference_sphere(P_chief, S_chief, axis_point=None, axis_dir=None):
    """Compute the exit-pupil reference sphere for a single chief ray.

    The reference sphere is centered on the chief ray's image point (P_chief)
    and has radius |P_xp - P_chief|, where P_xp is the chief ray's closest
    approach to the optical axis (the line through axis_point parallel to
    axis_dir).  For a centered coaxial system, the optical axis is the
    z-axis through the origin (the defaults).

    For tilted/decentered systems, supply axis_point and axis_dir explicitly,
    or compute P_xp via independent means (e.g., from a bundle of chief rays
    from different fields) and pass it directly to opd_from_raytrace.

    Parameters
    ----------
    P_chief : ndarray
        shape (3,).  Position of the chief ray, typically at the image plane.
        This becomes the center of the reference sphere.
    S_chief : ndarray
        shape (3,).  Direction cosines of the chief ray after the last surface.
    axis_point : iterable, optional
        a point on the optical axis (default: origin)
    axis_dir : iterable, optional
        direction of the optical axis (default: +z)

    Returns
    -------
    C, R, P_xp : ndarray, float, ndarray
        sphere center (=P_chief), radius, exit pupil center

    """
    if axis_point is None:
        axis_point = np.zeros(3, dtype=np.asarray(P_chief).dtype)
    if axis_dir is None:
        axis_dir = np.array([0., 0., 1.], dtype=np.asarray(P_chief).dtype)
    C = np.asarray(P_chief)
    P_xp = _closest_approach_on_axis(P_chief, S_chief,
                                     np.asarray(axis_point),
                                     np.asarray(axis_dir))
    R = np.sqrt(np.sum((P_xp - C) ** 2))
    return C, float(R), P_xp


def opd_from_raytrace(P_hist, S_hist, OPL_hist, P_img, P_xp,
                      n_image=1.0, chief_index=None):
    """Compute OPD on the exit-pupil reference sphere from a ray-trace.

    For each ray, the OPL is summed through the prescription, then extended
    along the final direction cosine until it intersects the reference sphere
    centered on P_img with radius |P_xp - P_img|.  OPD is reported relative
    to the chief ray (chief OPD = 0).

    The starting plane for OPL accumulation is whatever transverse plane the
    rays were launched on (e.g. z = const for a collimated fan).  Provided all
    rays in the bundle share that starting plane, the constant OPL offset
    cancels in the chief-relative subtraction.

    Parameters
    ----------
    P_hist : ndarray
        position history from raytrace, shape (jj+1, N, 3)
    S_hist : ndarray
        direction cosine history from raytrace, shape (jj+1, N, 3)
    OPL_hist : ndarray
        per-segment OPL history from raytrace, shape (jj+1, N).  OPL_hist[0]
        is zero by convention; OPL_hist[j+1] is the OPL of the segment ending
        at surface j.
    P_img : iterable
        image point — center of the reference sphere
    P_xp : iterable
        exit pupil center — sets the radius
    n_image : float
        index of refraction in image space (1=vacuum)
    chief_index : int, optional
        row index of the chief ray.  If None, defaults to N//2 which matches
        the convention used by raygen.generate_collimated_ray_fan.

    Returns
    -------
    opd : ndarray
        shape (N,).  Optical path difference at the reference sphere, in the
        same length units as the prescription.  Sign convention: chief == 0;
        rays whose OPL exceeds the chief's are positive.

    """
    P_img = np.asarray(P_img)
    P_xp = np.asarray(P_xp)
    R = np.sqrt(np.sum((P_xp - P_img) ** 2))

    P_last = P_hist[-1]
    S_last = S_hist[-1]
    Q, t = spencer_and_murty.intersect_reference_sphere(P_last, S_last, P_img, R)

    OPL_through = OPL_hist.sum(axis=0)
    OPL_total = OPL_through + n_image * t

    if chief_index is None:
        chief_index = OPL_total.shape[0] // 2

    return OPL_total - OPL_total[chief_index]


def locate_xp(P_chief, S_chief, P_img, P_sk):
    """Locate the exit pupil of a system.

    Defines the optical axis as the line through P_img and P_sk, then finds
    the point on that axis closest to the chief ray.  For a coaxial system
    this reduces to the standard z-axis intersection; for tilted/decentered
    systems the user should supply a meaningful axis pair (P_img, P_sk).

    Parameters
    ----------
    P_chief : ndarray
        any point on the chief ray (e.g. final position at the image plane)
    S_chief : ndarray
        chief ray direction cosines (after the last surface)
    P_img : iterable
        image-side point on the optical axis (commonly the image point)
    P_sk : iterable
        a second axis point (commonly the last optical surface vertex —
        NOT the image plane).  Must differ from P_img.

    Returns
    -------
    ndarray
        position of the exit pupil (X,Y,Z)

    """
    S_axis = _establish_axis(np.asarray(P_img), np.asarray(P_sk))
    return _closest_approach_on_axis(P_chief, S_chief, np.asarray(P_img), S_axis)
