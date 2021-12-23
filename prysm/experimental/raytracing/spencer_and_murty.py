"""Spencer & Murty's General Ray-Trace algorithm."""
from prysm.mathops import np

from .surfaces import (
    STYPE_REFLECT,
    STYPE_REFRACT,
    STYPE_STOP,
    STYPE_SPACE,
)


SURFACE_INTERSECTION_DEFAULT_EPS = 1e-14
SURFACE_INTERSECTION_DEFAULT_MAXITER = 100


def newton_raphson_solve_s(P1, S, F, Fprime, s1=0,
                           eps=SURFACE_INTERSECTION_DEFAULT_EPS,
                           maxiter=SURFACE_INTERSECTION_DEFAULT_MAXITER):
    """Use Newton-Raphson iteration to solve for intersection between a ray and surface.

    Parameters
    ----------
    P1 : numpy.ndarray
        position (X1,Y1,Z1) at in the plane normal to the surface vertex
        Eq. 7 from Spencer & Murty, except we keep Z1 so we can utilize vector algebra
    S : numpy.ndarray
        (k,l,m) incident direction cosines
    F : callable of signature F(x,y) -> z
        a function  which returns the surface sag at point x, y
    Fprime : callable of signature F'(x,y) -> Fx, Fy
        a function  which returns the cartesian derivatives of the sag at point x, y
    s1 : float
        initial guess for the length along the ray from (X1, Y1, 0) to reach the surface
    eps : float
        tolerance for convergence of Newton's method
    maxiter : int
        maximum number of iterations to allow

    Returns
    -------
    Pj, r : numpy.ndarray, numpy.ndarray
        final position of the ray intersection, and the surface normal at that point

    """
    P1 = np.asarray(P1)
    S = np.asarray(S)
    sj = s1
    for j in range(maxiter):
        Pj = sj * S + P1
        Xj, Yj, Zj = Pj
        Fj = Zj - F(Xj, Yj)
        r = Fprime(Xj, Yj)
        Fpj = np.dot(r, S)
        sjp1 = sj - Fj / Fpj

        delta = abs(sjp1 - sj)
        sj = sjp1
        if delta < eps:
            break
    # TODO: handle non-convergence
    return Pj, r


def intersect(P0, S, F, Fprime, s1=0,
              eps=SURFACE_INTERSECTION_DEFAULT_EPS,
              maxiter=SURFACE_INTERSECTION_DEFAULT_MAXITER):
    """Find the intersection of a ray and a surface.

    Parameters
    ----------
    P0 : numpy.ndarray
        position of the ray, in local coordinates (but Z not necessarily zero)
        Eq. 3 Spencer & Murty
    S : numpy.ndarray
        (k,l,m) incident direction cosines
    F : callable of signature F(x,y) -> z
        a function  which returns the surface sag at point x, y
    Fprime : callable of signature F'(x,y) -> Fx, Fy
        a function  which returns the cartesian derivatives of the sag at point x, y
    s1 : float
        initial guess for the length along the ray from (X1, Y1, 0) to reach the surface
    eps : float
        tolerance for convergence of Newton's method
    maxiter : int
        maximum number of iterations to allow

    Returns
    -------
    Pj, r : numpy.ndarray, numpy.ndarray
        final position of the ray intersection, and the surface normal at that point

    """
    # go to z=0
    Z0 = P0[2]
    m = S[2]
    s0 = -Z0/m
    # Eq. 7, in vector form (extra computation on Z is cheaper than breaking apart P and S)
    P1 = P0 + np.dot(s0, S)
    # then use newton's method to find and go to the intersection
    return newton_raphson_solve_s(P1, S, F, Fprime, s1, eps, maxiter)


def transform_to_local_coords(XYZ, P, S, R=None):
    """Transform the coordinates XYZ to local coordinates about P, plausibly rotated by R.

    Parameters
    ----------
    XYZ : numpy.ndarray
        "world" coordinates [X,Y,Z]
    P : numpy.ndarray of shape (3,)
        point defining the origin of the local coordinate frame, [X0,Y0,Z0]
    S : numpy.ndarray
        (k,l,m) incident direction cosines
    R : numpy.ndarray of shape (3,3)
        rotation matrix to apply, if the surface is tilted

    Returns
    -------
    numpy.ndarray, numpy.ndarray
        rotated XYZ coordinates, rotated direction cosines

    """
    XYZ2 = XYZ - P
    if R is not None:
        XYZ2 = np.matmul(R, XYZ2)
        S = np.matmul(R, S)

    return XYZ2, S


def refract(n, nprime, S, r, gamma1=None, eps=1e-14, maxiter=100):
    """Use Newton-Raphson iteration to solve Snell's law for the exitant direction cosines.

    Parameters
    ----------
    n : float
        preceeding index of refraction
    nprime : float
        following index of refraction
    S : numpy.ndarray
        length 3 vector containing the input direction cosines
    r : numpy.ndarray
        length 3 vector containing the surface normals (Fx, Fy, 1)
    gamma1 : float
        guess for gamma, if none -b/2a as in Eq. 44
    eps : float
        tolerance for convergence of Newton's method
    maxiter : int
        maximum number of iterations to allow

    Returns
    -------
    numpy.ndarray
        Sprime, a length 3 vector containing the exitant direction cosines

    """
    mu = n/nprime
    musq = mu * mu
    if len(r) == 2:
        r = np.array([*r, 1])
    rnorm = (r*r).sum()

    a = mu * np.dot(S, r) / rnorm
    b = (musq - 1) / rnorm
    if gamma1 is None:
        gamma1 = -b/(2*a)

    gammaj = gamma1
    for j in range(maxiter):
        # V(gamma)     = Gamma^2 + 2aGamma + b
        # V(gamma_n+1) = 2(Gamman + a)
        # Gamma_n+1 = (Gamman^2 - b)/(2*(Gamman + a))
        gammajp1 = (gammaj * gammaj - b)/(2*(gammaj + a))
        delta = abs(gammajp1 - gammaj)
        gammaj = gammajp1
        if delta < eps:
            break

    # now S' = mu * S + Gamma * r
    Sprime = mu * S + gammaj * r
    return Sprime


def reflect(S, r):
    """Reflect a ray off of a surface.

    Parameters
    ----------
    S : numpy.ndarray
        length 3 vector containing the input direction cosines
    r : numpy.ndarray
        length 3 vector containing the surface normals (Fx, Fy, 1)

    Returns
    -------
    numpy.ndarray
        Sprime, a length 3 vector containing the exitant direction

    """
    # TODO: can we avoid the rnorm calculation here?
    rnorm = (r*r).sum()
    # paragraph above Eq. 45, mu=1
    # and see that definition of a including
    # mu=1 does not require multiply by mu (1)
    cosI = np.dot(S, r) / rnorm
    return S - 2 * cosI * r


def raytrace(surfaces, P, S, wvl, n_ambient=1):
    """Perform a raytrace through a sequence of surfaces.

    Notes
    -----
    A ray originating "at infinity" would have
    P = [Px, Py, -1e99]
    S = [0, 0, 1] # propagating in the +z direction
    though the value of P is not so important, since S defines the ray as moving in the +z direction only

    Implementation Notes
    --------------------
    See Spencer & Murty, General Ray-Tracing Procedure JOSA 1961

    Steps (I, II, III, IV) utilize the functions:
    I   -> transform_to_local_coords
    II  -> newton_raphson_solve_s
    III -> reflect or refract
    IV  -> NOT IMPLEMENTED

    Parameters
    ----------
    surfaces : iterable
        the surfaces to trace through;
        a surface is defined by the interface:
        surf.F(x,y) -> z sag
        surf.Fp(x,y) -> (Fx, Fy, 1) derivatives (with S&M convention for 1 in z)
        surf.typ in {STYPE}
        surf.P, surface global coordinates, [X,Y,Z]
        surf.R, surface rotation matrix (may be None)
        surf.n(wvl) -> refractive index (wvl in um)
    P : numpy.ndarray
        position (X0,Y0,Z0) at the outset of the raytrace
    S : numpy.ndarray
        (k,l,m) starting direction cosines
    wvl : float
        wavelength of light, um
    n_ambient : float
        ambient index of refraction (1=vacuum)

    Returns
    -------
    P_hist, S_hist
        position history and direction cosine history

    """
    P_hist = [P]
    S_hist = [S]
    Pj = P
    Sj = S
    nj = n_ambient
    for surf in surfaces:
        # for space surfaces, simply propagate the rays
        if surf.typ == STYPE_SPACE:
            Sjp1 = Sj
            Pjp1 = Pj + np.dot(surf.P, Sj)  # P is separation along the ray (~= surface sep)
            P_hist.append(Pjp1)
            S_hist.append(Sjp1)
            Pj, Sj = Pjp1, Sjp1
            continue

        # I - transform from global to local coordinates
        P0, Sj = transform_to_local_coords(Pj, surf.P, Sj, surf.R)
        # II - find ray intersection
        Pj, r = intersect(P0, Sj, surf.sag, surf.normal)
        # III - reflection or refraction
        if surf.typ == STYPE_REFLECT:
            Sjp1 = reflect(Sj, r)
        elif surf.typ == STYPE_REFRACT:
            nprime = surf.n(wvl)
            Sjp1 = refract(nj, nprime, Sj, r)
            nj = nprime

        # IV - back to world coordinates
        if surf.R is None:
            Rt = None
        else:
            # transformation matrix has inverse which is its transpose
            Rt = surf.R.T
        Pjp1, Sjp1 = transform_to_local_coords(Pj, -surf.P, Sjp1, Rt)
        P_hist.append(Pjp1)
        S_hist.append(Sjp1)
        Pj, Sj = Pjp1, Sjp1

    return P_hist, S_hist
