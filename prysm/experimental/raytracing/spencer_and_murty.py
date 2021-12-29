"""Spencer & Murty's General Ray-Trace algorithm."""

# don't uncomment this line, this is a very obscure import
# not using it at the moment for GPU compatability
# from numpy.core.umath_tests import inner1d

from prysm.mathops import np

from .surfaces import (
    STYPE_REFLECT,
    STYPE_REFRACT,
    STYPE_STOP,
    STYPE_SPACE,
)


SURFACE_INTERSECTION_DEFAULT_EPS = 1e-14
SURFACE_INTERSECTION_DEFAULT_MAXITER = 100


def _multi_dot(a, b):
    """Dot product between a and b along the last (batch) dimension.

    Implementation will change over time to track the fastest way to do this
    with numpy. (maybe)

    There is no BLAS level 1/2/3 function for a batch of dot products.

    For a (1024*1024)*3, aka 1 million dot batch, the fastest function below
    takes 4.23 ms on a dual channel laptop (~40GB/s bandwidth from RAM).

    The dot product is simply sum += a[i]*b[i], which touches three values for
    each element of the input array, i.e.
    sum = 0
    sum += a[0] + b[0]
    sum += a[1] + b[1]
    sum += a[2] + b[2]

    It also performs one flop (floating-point operation) per element.

    with 6,291,456 elements and eight bytes per element, this is 50,331,648 bytes
    of computation in 4.23 ms, 11,898,734,751 (about 12GB/sec)

    So, this can be made faster by using a few threads, but those threads must not
    perform extra copies, since we are near the memory bandwidth limit of the system

    But, in a gist
    https://gist.github.com/brandondube/43ab9e9f173252f5a97e0c0d5c3ca54f

    with batch size 1 million (einsum is not the fastest below)
    no parallelism - 7.39 ms ± 424 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    thread pool 1  - 8.8 ms ± 259 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    thread pool 2  - 5.23 ms ± 68.5 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    thread pool 3  - 5.77 ms ± 177 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

    best case 30% speedup - not worth it

    rule of thumb, intel CPU can start two floating point operations per clock per core
    ~= 4 billion clocks per second ~= 8 billion flops/sec/core

    we need 6.3M flops for our example batch size, and the calc took 4.3 ms, so
    1,463,129,302 flops/sec were used (~= 25% of one CPU core)

    so, the task was memory bandwidth limited, and we go faster with multiple
    threads only because of some quirk of intel's memory controller and prefetch
    semantics.  But >10x faster is not living in reality.  Maybe on a system
    with vast memory bandwidth (say, 4 socket xeon -- 800GB/sec).  But that's
    $40k of CPUs and one A40 GPU does that for $5k, so why bother.

    """
    return np.einsum('ij,ij->i', a, b)
    # return np.matmul(a[:,None,:], b[:,:,None])
    # return np.sum(a*b, axis=1)
    # return inner1d(a, b)


def newton_raphson_solve_s(P1, S, F, Fprime, s1=0.0,
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
    # need one sj for each ray, but sj likely starts as a scalar
    # so, make sure it's at least 1D, then broadcast it to the number of rays
    # finally, add a size 1 final dim to make multiply broadcast rules happy
    nrays = P1.shape[0]
    sj = np.atleast_1d(s1)
    sj = np.broadcast_to(sj, (nrays,)).copy()  # copy is needed to make writeable
    sj = sj.astype(np.float64)
    # the Pj and r to be returned; we keep these three data structures around
    # so they can be adjusted within the loop
    Pj_out = np.empty_like(P1)
    r_out = np.empty((nrays, 3), dtype=P1.dtype)
    mask = np.arange(nrays)
    for j in range(maxiter):
        sj_mask = sj[mask]
        sj_bcast = sj_mask[:, np.newaxis]
        S_mask = S[mask]
        Pj = P1[mask] + sj_bcast * S_mask
        Xj = Pj[..., 0]
        Yj = Pj[..., 1]
        Zj = Pj[..., 2]
        Fj = Zj - F(Xj, Yj)
        r = Fprime(Xj, Yj)
        # r*S.sum(axis=1) == np.dot(r,S)
        Fpj = (r * S_mask).sum(axis=1)

        sjp1 = sj_mask - Fj / Fpj

        delta = abs(sjp1 - sj_mask)

        # the final piece of the pie, the iterations will not end
        # for all rays at once, we need a way to end each ray independently
        # solution: keep an index array, which is which elements of Pj and r
        # are being worked on, initialized to all rays
        # modify the index array by masking on delta < eps and assign into
        # Pj and r based on that
        rays_which_converged = delta < eps
        sj[mask] = sjp1
        insert_mask = mask[rays_which_converged]
        if insert_mask.size != 0:
            Pj_out[insert_mask] = Pj[rays_which_converged]
            r_out[insert_mask] = r[rays_which_converged]
            # update the mask for the next iter to only those rays which
            # did not converge
            mask = mask[~rays_which_converged]
            if mask.shape[0] == 0:
                break  # all rays converged

    # TODO: handle non-convergence
    return Pj_out, r_out


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
    # batch support -- ellipsis skip any early dimensions, then replace
    # dot with a multiply
    P0, S = np.atleast_2d(P0, S)
    # go to z=0
    Z0 = P0[..., 2]
    m = S[..., 2]
    s0 = -Z0/m
    # Eq. 7, in vector form (extra computation on Z is cheaper than breaking apart P and S)
    # the newaxis on s0 turns (N,) -> (N,1) so that multiply's broadcast rules work
    P1 = P0 + s0[:, np.newaxis] * S
    # P1 is (N,3)
    # then use newton's method to find and go to the intersection
    return newton_raphson_solve_s(P1, S, F, Fprime, s1, eps, maxiter)


def transform_to_global_coords(XYZ, P, S, R=None):
    """Transform the coordiantes XYZ from local coordinates about P back to global coordinates.

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
    if R is not None:
        XYZ, S = np.atleast_2d(XYZ, S)
        XYZ = np.matmul(R, XYZ[..., np.newaxis]).squeeze(-1)
        S = np.matmul(R, S[..., np.newaxis]).squeeze(-1)

    XYZ = XYZ + P
    return XYZ, S


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
        XYZ2, S = np.atleast_2d(XYZ2, S)
        # in regular matmul, 3x3 @ (3,) has a 1 appended to the dimension
        # of the second array to make it into a column vector
        # for batch compatability, we do that manually
        XYZ2 = np.matmul(R, XYZ2[..., np.newaxis]).squeeze(-1)
        S = np.matmul(R, S[..., np.newaxis]).squeeze(-1)

    return XYZ2, S


def refract(n, nprime, S, r):
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

    Returns
    -------
    numpy.ndarray
        Sprime, a length 3 vector containing the exitant direction cosines

    """
    mu = n/nprime
    musq = mu * mu
    # r*S.sum(axis=1) == np.dot(r,S)
    cosI = _multi_dot(r, S)
    cosIsq = cosI * cosI
    # the inline newaxis-es are terrible for readability, but serve a performance purpose
    # broadcast the square root to 2D, so that fewer very expensive sqrt ops are done
    # then, in the second term, broadcast cosI for compatability with S and r
    # since it is needed there
    first_term = np.sqrt(1 - musq * (1 - cosIsq))[:, np.newaxis] * r
    second_term = mu * (S - cosI[:, np.newaxis] * r)
    return first_term + second_term


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
    # at least 2D turns (3,) -> (1,3) where 1 = batch reflect count
    # this allows us to use the same code for vector operations on many
    # S, r or on one S, r
    S, r = np.atleast_2d(S, r)
    rnorm = _multi_dot(r, r)

    # paragraph above Eq. 45, mu=1
    # and see that definition of a including
    # mu=1 does not require multiply by mu (1)
    # equivalent code for single ray: cosI = np.dot(S, r) / rnorm
    cosI = _multi_dot(S, r) / rnorm

    # another trick, cosI scales each vector, we need to give cosI proper dimensionality
    # since it is 1D and r is 2D
    # (2,) -> (2,1)
    # where 2 = number of parallel rays being traced
    cosI = cosI[:, np.newaxis]
    # return will be (2, 3) where 2 = number of parallel rays
    # other operations in the ray-trace procedure would view the array up to 2D
    # even if it were 1D, so we do not squeeze here
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
    jj = len(surfaces)
    P_hist = np.empty((jj+1, *P.shape), dtype=P.dtype)
    S_hist = np.empty((jj+1, *S.shape), dtype=P.dtype)
    Pj = P
    Sj = S
    P_hist[0] = P
    S_hist[0] = S
    nj = n_ambient
    for j, surf in enumerate(surfaces):
        # for space surfaces, simply propagate the rays
        if surf.typ == STYPE_SPACE:
            Sjp1 = Sj
            Pjp1 = Pj + np.dot(surf.P, Sj)  # P is separation along the ray (~= surface sep)
            P_hist[j+1] = Pjp1
            S_hist[j+1] = Sjp1
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
        Pjp1, Sjp1 = transform_to_global_coords(Pj, surf.P, Sjp1, Rt)
        P_hist[j+1] = Pjp1
        S_hist[j+1] = Sjp1
        Pj, Sj = Pjp1, Sjp1

    return P_hist, S_hist
