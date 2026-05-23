"""3D line/ray geometry helpers used by raytracing routines."""

from prysm.mathops import np


def normalize_vector(v, axis=-1):
    """Return v scaled to unit length along axis."""
    v = np.asarray(v)
    return v / np.linalg.norm(v, axis=axis, keepdims=True)


def unit_vector_between(P1, P2):
    """Return the unit vector pointing from P1 to P2.

    Parameters
    ----------
    P1 : array-like
        starting point
    P2 : array-like
        ending point

    Returns
    -------
    ndarray
        unit vector with the same shape as P2 - P1

    """
    diff = np.asarray(P2) - np.asarray(P1)
    return normalize_vector(diff, axis=-1)


def line_intersection_params(P1, S1, P2, S2):
    """Return least-squares line parameters for two 3D rays.

    Solves [S1, -S2] @ [t1, t2] = P2 - P1 in the least-squares sense,
    yielding the parameters along each ray that minimize the squared
    distance between the two lines.

    Parameters
    ----------
    P1 : ndarray
        origin of ray 1, length-3
    S1 : ndarray
        direction of ray 1, length-3
    P2 : ndarray
        origin of ray 2, length-3
    S2 : ndarray
        direction of ray 2, length-3

    Returns
    -------
    ndarray
        length-2 vector (t1, t2) of parameters along S1 and S2

    """
    d = P2 - P1
    a = np.dot(S1, S1)
    b = np.dot(S1, S2)
    c = np.dot(S2, S2)
    e = np.dot(S1, d)
    f = np.dot(S2, d)
    denom = a * c - b * b
    if abs(denom) < 1e-30:
        # parallel rays; fall back to projecting d onto S1
        return np.asarray([e / a, 0.0])
    t1 = (c * e - b * f) / denom
    t2 = (b * e - a * f) / denom
    return np.asarray([t1, t2])


def closest_point_on_line_to_line(P, S, axis_point, axis_dir):
    """Point on an axis line closest to another 3D line.

    Parameters
    ----------
    P : array-like
        origin of the query line, length-3
    S : array-like
        direction of the query line, length-3 (need not be unit)
    axis_point : array-like
        a point on the axis line, length-3
    axis_dir : array-like
        direction of the axis line, length-3 (normalized internally)

    Returns
    -------
    ndarray
        length-3 point on the axis line that is closest to the query line;
        if the two lines are parallel, returns the foot of perpendicular
        from P onto the axis line

    """
    A = np.asarray(P)
    Sc = np.asarray(S)
    B = np.asarray(axis_point)
    Sa = normalize_vector(axis_dir, axis=-1)
    w = A - B
    a = np.dot(Sc, Sc)
    b = np.dot(Sc, Sa)
    c = np.dot(Sa, Sa)
    d = np.dot(Sc, w)
    e = np.dot(Sa, w)
    denom = a * c - b * b
    if abs(denom) < 1e-30:
        t = e / c
        return B + t * Sa
    t = (a * e - b * d) / denom
    return B + t * Sa
