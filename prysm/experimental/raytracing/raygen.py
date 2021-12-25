"""Ray (grid/fan) generation routines."""

from prysm.mathops import np
from prysm.coordinates import make_rotation_matrix, polar_to_cart


def concat_rayfans(*rayfans):
    """Merge N rayfans for a single batch trace.

    Parameters
    ----------
    rayfans : tuple of (P, S)
        the result of any of the generate_* functions in this file.

    Returns
    -------
    numpy.ndarray, numpy.ndarray
        concatonated P, S

    """
    ps = []
    ss = []
    for (p, s) in rayfans:
        ps.append(p)
        ss.append(s)

    Ps = np.vstack(ps)
    Ss = np.vstack(ss)
    return Ps, Ss


def generate_collimated_ray_fan(nrays, maxr, z=0, minr=None, azimuth=90,
                                yangle=0, xangle=0,
                                distribution='uniform', aim_at=None):
    """Generate a 1D fan of rays.

    Colloquially, an extended field in Y for an object at inf is represented by a ray fan with yangle != 0.

    Parameters
    ----------
    nrays : int
        the number of rays in the fan
    maxr : float
        maximum radial value of the fan
    z : float
        z position for the ray fan
    minr : float, optional
        minimum radial value of the fan, -maxr if None
    azimuth: float
        angle in the XY plane, degrees.  0=X ray fan, 90=Y ray fan
    yangle : float
        propagation angle of the rays with respect to the Y axis, clockwise
    xangle : float
        propagation angle of the rays with respect to the X axis, clockwise
    distribution : str, {'uniform', 'random', 'cheby'}
        The distribution to use when placing the rays
        a uniform distribution has rays which are equally spaced from minr to maxr,
        random has rays randomly distributed in minr and maxr, while cheby has the
        Cheby-Gauss-Lobatto roots as its locations from minr to maxr
    aim_at : numpy.ndarray or float
        position [X,Y,Z] aim the rays such that the gut ray of the fan,
        when propagated in an open medium, hits (aim_at)

        if a float, interpreted as if x=0,y=0, z=aim_at

        This argument mimics ray aiming in commercial optical design software, and
        is used in conjunction with xangle/yangle to form off-axis ray bundles which
        properly go through the center of the stop

    Returns
    -------
    numpy.ndarray, numpy.ndarray
        "P" and "S" variables, positions and direction cosines of the rays

    """
    distribution = distribution.lower()
    if minr is None:
        minr = -maxr
    S = np.array([0, 0, 1])
    R = make_rotation_matrix((0, yangle, -xangle))
    S = np.matmul(R, S)
    # need to see a copy of S for each ray, -> add empty dim and broadcast
    S = S[np.newaxis, :]
    S = np.broadcast_to(S, (nrays, 3))

    # now generate the radial part of P
    if distribution == 'uniform':
        r = np.linspace(minr, maxr, nrays)
    elif distribution == 'random':
        r = np.random.uniform(low=minr, high=maxr, size=nrays)

    t = np.broadcast_to(np.radians(azimuth), r.shape)
    x, y = polar_to_cart(r, t)
    z = np.broadcast_to(z, x.shape)
    xyz = np.stack([x, y, z], axis=1)
    return xyz, S


def generate_collimated_rect_ray_grid(nrays, maxx, z=0, minx=None, maxy=None, miny=None, yangle=0, xangle=0, distribution='uniform'):
    """Generate a 2D grid of rays on a rectangular basis.

    Parameters
    ----------
    nrays : int
        the number of rays in each axis of the fan, there will be nrays^2 rays total
    maxx : float
        maximum x coordinate of the fan
    z : float
        z position for the ray fan
    minx : float, optional
        minimum x coordinate of the fan, -maxx if None
    maxy : float
        maximum y coordinate of the fan, maxx if None
    miny : float, optional
        minimum y coordinate of the fan, -minx if None
    yangle : float
        propagation angle of the rays with respect to the Y axis, clockwise
    xangle : float
        propagation angle of the rays with respect to the X axis, clockwise
    distribution : str, {'uniform', 'random', 'cheby'}
        The distribution to use when placing the rays
        a uniform distribution has rays which are equally spaced from minr to maxr,
        random has rays randomly distributed in minr and maxr, while cheby has the
        Cheby-Gauss-Lobatto roots as its locations from minr to maxr

    Returns
    -------
    numpy.ndarray, numpy.ndarray
        "P" and "S" variables, positions and direction cosines of the rays

    """
    distribution = distribution.lower()
    if minx is None:
        minx = -maxx
    if maxy is None:
        maxy = maxx
    if miny is None:
        miny = minx

    S = np.array([0, 0, 1])
    R = make_rotation_matrix((0, yangle, -xangle))
    S = np.matmul(R, S)
    # need to see a copy of S for each ray, -> add empty dim and broadcast
    S = S[np.newaxis, :]
    S = np.broadcast_to(S, (nrays*nrays, 3))

    # now generate the x and y fans
    if distribution == 'uniform':
        x = np.linspace(minx, maxx, nrays)
        y = np.linspace(miny, maxy, nrays)
        xx, yy = np.meshgrid(x, y)
        xx = xx.ravel()
        yy = yy.ravel()
    elif distribution == 'random':
        x = np.random.uniform(low=minx, high=maxx, size=nrays)
        y = np.random.uniform(low=miny, high=maxy, size=nrays)
        xx, yy = np.meshgrid(x, y)
        xx = xx.ravel()
        yy = yy.ravel()

    z = np.broadcast_to(z, xx.shape)
    xyz = np.stack([xx, yy, z], axis=1)
    return xyz, S

# TODO: cheby-gauss-lobatto-forbes circular spiral, random spiral
