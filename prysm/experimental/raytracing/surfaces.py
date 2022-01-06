"""Surface types and calculus."""

from prysm.mathops import np
from prysm.conf import config
from prysm.coordinates import cart_to_polar, make_rotation_matrix
from prysm.polynomials.qpoly import compute_z_zprime_Q2d
from prysm.polynomials import hermite_He_sequence, lstsq, mode_1d_to_2d


def find_zero_indices_2d(x, y, tol=1e-8):
    """Find the (y,x) indices into x and y where x==y==0."""
    # assuming we're FFT-centered, we will never do the ifs
    # this probably blows up if zero is not in the array
    lookup = tuple(s//2 for s in x.shape)
    x0 = x[lookup]
    if x0 > tol:
        lookup2 = (lookup[0], lookup[1]+1)
        x1 = x[lookup2]
        dx = x1-x0
        shift_samples = (x0 / dx)
        lookup = (lookup[0], lookup[1]+shift_samples)
    y0 = y[lookup]
    if y0 > tol:
        lookup2 = (lookup[0]+1, lookup[1])
        y1 = x[lookup2]
        dy = y1-y0
        shift_samples = (y0 / dy)
        lookup = (lookup[0]+shift_samples, lookup[1])

    return lookup


def fix_zero_singularity(arr, x, y, fill='xypoly', order=2):
    """Fix a singularity at the origin of arr by polynomial interpolation.

    Parameters
    ----------
    arr : numpy.ndarray
        array of dimension 2 to modify at the origin (x==y==0)
    x : numpy.ndarray
        array of dimension 2 of X coordinates
    y : numpy.ndarray
        array of dimension 2 of Y coordinates
    fill : str, optional, {'xypoly'}
        how to fill.  Not used/hard-coded to X/Y polynomials, but made an arg
        today in case it may be added future for backwards compatibility
    order : int
        polynomial order to fit

    Returns
    -------
    numpy.ndarray
        arr (modified in-place)

    """
    zloc = find_zero_indices_2d(x, y)
    min_y = zloc[0]-order
    max_y = zloc[0]+order+1
    min_x = zloc[1]-order
    max_x = zloc[1]+order+1
    # newaxis schenanigans to get broadcasting right without
    # meshgrid
    ypts = np.arange(min_y, max_y)[:, np.newaxis]
    xpts = np.arange(min_x, max_x)[np.newaxis, :]
    window = arr[ypts, xpts].copy()
    c = [s//2 for s in window.shape]
    window[c] = np.nan
    # no longer need xpts, ypts
    # really don't care about fp64 vs fp32 (very small arrays)
    xpts = xpts.astype(float)
    ypts = ypts.astype(float)
    # use Hermite polynomials as
    # XY polynomial-like basis orthogonal
    # over the infinite plane
    # H0 = 0
    # H1 = x
    # H2 = x^2 - 1, and so on
    ns = np.arange(order+1)
    xbasis = hermite_He_sequence(ns, xpts)
    ybasis = hermite_He_sequence(ns, ypts)
    xbasis = [mode_1d_to_2d(mode, xpts, ypts, 'x') for mode in xbasis]
    ybasis = [mode_1d_to_2d(mode, xpts, ypts, 'y') for mode in ybasis]
    basis_set = np.asarray([*xbasis, *ybasis])
    coefs = lstsq(basis_set, window)
    projected = np.dot(basis_set[:, c[0], c[1]], coefs)
    arr[zloc] = projected
    return arr


def surface_normal_from_cylindrical_derivatives(fp, ft, r, t):
    """Use polar derivatives to compute Cartesian surface normals.

    Parameters
    ----------
    fp : numpy.ndarray
        derivative of f w.r.t. r
    ft : numpy.ndarray
        derivative of f w.r.t. t
    r : numpy.ndarray
        radial coordinates
    t : numpy.ndarray
        azimuthal coordinates

    Returns
    -------
    numpy.ndarray, numpy.ndarray
        x, y derivatives; will contain a singularity where r=0,
        see fix_zero_singularity

    """
    cost = np.cos(t)
    sint = np.sin(t)
    x = fp * cost - 1/r * ft * sint
    y = fp * sint + 1/r * ft * cost
    return x, y


def surface_normal_from_cartesian_derivatives(fx, fy, r, t):
    """Use Cartesian derivatives to compute polar surface normals.

    Parameters
    ----------
    fx : numpy.ndarray
        derivative of f w.r.t. x
    fy : numpy.ndarray
        derivative of f w.r.t. y
    r : numpy.ndarray
        radial coordinates
    t : numpy.ndarray
        azimuthal coordinates

    Returns
    -------
    numpy.ndarray, numpy.ndarray
        r, t derivatives; will contain a singularity where r=0,
        see fix_zero_singularity

    """
    cost = np.cos(t)
    sint = np.sin(t)
    onebyr = 1/r
    r = fx * cost + fy * sint
    t = fx * -sint / onebyr + fy * cost / onebyr
    return r, t


def product_rule(u, v, du, dv):
    """The product rule of calculus, d/dx uv = u dv v du."""
    return u * dv + v * du


def phi_spheroid(c, k, rhosq):
    """'phi' for a spheroid.

    phi = sqrt(1 - c^2 rho^2)

    Parameters
    ----------
    c : float
        curvature, reciprocal radius of curvature
    k : float
        kappa, conic constant
    rhosq : numpy.ndarray
        squared radial coordinate (non-normalized)

    Returns
    -------
    numpy.ndarray
        phi term

    """
    csq = c * c
    return np.sqrt(1 - (1 + k) * csq * rhosq)


def der_direction_cosine_spheroid(c, k, rho, rhosq=None, phi=None):
    """Derivative term needed for the product rule and Q type aspheres.

    sag z(rho) = 1/phi * (weighted sum of Q polynomials)

    The return of this function is the derivative of 1/phi required
    for completing the product rule on the surface's derivative.

    Parameters
    ----------
    c : float
        curvature, reciprocal radius of curvature
    k : float
        kappa, conic constant
    rho : numpy.ndarray
        radial coordinate (non-normalized)
    rhosq : numpy.ndarray
        squared radial coordinate (non-normalized)
        rho ** 2 if None
    phi : numpy.ndarray, optional
        (1 - c^2 r^2)^.5
        computed if not provided
        many surface types utilize phi; its computation can be
        de-duplicated by passing the optional argument

    Returns
    -------
    numpy.ndarray
        d/drho of (1/phi)

    """
    csq = c * c
    if rhosq is None:
        rhosq = rho * rho
    if phi is None:
        phi = phi_spheroid(c, k, rhosq)

    num = -csq * (k-1) * rho
    den = phi * phi * phi
    return num / den


def sphere_sag(c, rhosq, phi=None):
    """Sag of a spherical surface.

    Parameters
    ----------
    c : float
        surface curvature
    rhosq : numpy.ndarray
        radial coordinate squared
        e.g. for a 15 mm half-diameter optic,
        rho = 0 .. 15
        rhosq = 0 .. 225
        there is no requirement on rectilinear sampling or array
        dimensionality
    phi : numpy.ndarray, optional
        (1 - c^2 r^2)^.5
        computed if not provided
        many surface types utilize phi; its computation can be
        de-duplicated by passing the optional argument

    Returns
    -------
    numpy.ndarray
        surface sag

    """
    if phi is None:
        csq = c * c
        phi = np.sqrt(1 - csq * rhosq)

    return (c * rhosq) / (1 + phi)


def sphere_sag_der(c, rho, phi=None):
    """Derivative of the sag of a spherical surface.

    Parameters
    ----------
    c : float
        surface curvature
    rho : numpy.ndarray
        radial coordinate
        e.g. for a 15 mm half-diameter optic,
        rho = 0 .. 15
        there is no requirement on rectilinear sampling or array
        dimensionality
    phi : numpy.ndarray, optional
        (1 - c^2 r^2)^.5
        computed if not provided
        many surface types utilize phi; its computation can be
        de-duplicated by passing the optional argument

    Returns
    -------
    numpy.ndarray
        derivative of surface sag

    """
    if phi is None:
        csq = c ** 2
        rhosq = rho * rho
        phi = np.sqrt(1 - csq * rhosq)
    return (c * rho) / phi


def conic_sag(c, kappa, rhosq, phi=None):
    """Sag of a spherical surface.

    Parameters
    ----------
    c : float
        surface curvature
    kappa : float
        conic constant
    rhosq : numpy.ndarray
        radial coordinate squared
        e.g. for a 15 mm half-diameter optic,
        rho = 0 .. 15
        rhosq = 0 .. 225
        there is no requirement on rectilinear sampling or array
        dimensionality
    phi : numpy.ndarray, optional
        (1 - (1+kappa) c^2 r^2)^.5
        computed if not provided
        many surface types utilize phi; its computation can be
        de-duplicated by passing the optional argument

    Returns
    -------
    numpy.ndarray
        surface sag

    """
    if phi is None:
        csq = c * c
        phi = np.sqrt(1 - (1+kappa) * csq * rhosq)

    return (c * rhosq) / (1 + phi)


def conic_sag_der(c, kappa, rho, phi=None):
    """Sag of a spherical surface.

    Parameters
    ----------
    c : float
        surface curvature
    kappa : float
        conic constant, 0=sphere, 1=parabola, etc
    rho : numpy.ndarray
        radial coordinate
        e.g. for a 15 mm half-diameter optic,
        rho = 0 .. 15
        there is no requirement on rectilinear sampling or array
        dimensionality
    phi : numpy.ndarray, optional
        (1 - (1+kappa) c^2 r^2)^.5
        computed if not provided
        many surface types utilize phi; its computation can be
        de-duplicated by passing the optional argument

    Returns
    -------
    numpy.ndarray
        surface sag

    """
    if phi is None:
        csq = c ** 2
        rhosq = rho * rho
        phi = np.sqrt(1 - (1+kappa) * csq * rhosq)

    return (c * rho) / phi


def off_axis_conic_sag(c, kappa, r, t, dx, dy=0):
    """Sag of an off-axis conicoid.

    Parameters
    ----------
    c : float
        axial curvature of the conic
    kappa : float
        conic constant
    r : numpy.ndarray
        radial coordinate, where r=0 is centered on the off-axis section
    t : numpy.ndarray
        azimuthal coordinate
    dx : float
        shift of the surface in x with respect to the base conic vertex,
        mutually exclusive to dy (only one may be nonzero)
        use dx=0 when dy != 0
    dy : float
        shift of the surface in y with respect to the base conic vertex

    Returns
    -------
    numpy.ndarray
        surface sag, z(x,y)

    """
    if dy != 0 and dx != 0:
        raise ValueError('only one of dx/dy may be nonzero')

    if dx != 0:
        s = dx
        oblique_term = 2 * s * r * np.cos(t)
    else:
        s = dy
        oblique_term = 2 * s * r * np.sin(t)

    aggregate_term = r * r + oblique_term + s * s
    num = c * aggregate_term
    csq = c * c
    den = 1 + np.sqrt(1 - (1 + kappa) * csq * aggregate_term)
    return num / den


def off_axis_conic_der(c, kappa, r, t, dx, dy=0):
    """Radial and azimuthal derivatives of an off-axis conic.

    Parameters
    ----------
    c : float
        axial curvature of the conic
    kappa : float
        conic constant
    r : numpy.ndarray
        radial coordinate, where r=0 is centered on the off-axis section
    t : numpy.ndarray
        azimuthal coordinate
    dx : float
        shift of the surface in x with respect to the base conic vertex,
        mutually exclusive to dy (only one may be nonzero)
        use dx=0 when dy != 0
    dy : float
        shift of the surface in y with respect to the base conic vertex

    Returns
    -------
    numpy.ndarray, numpy.ndarray
        d/dr(z), d/dt(z)

    """
    if dy != 0 and dx != 0:
        raise ValueError('only one of dx/dy may be nonzero')

    cost = np.cos(t)
    sint = np.sin(t)
    if dx != 0:
        s = dx
        oblique_term = 2 * s * r * cost
        ddr_oblique = 2 * r + 2 * s * cost
        # I accept the evil in writing this the way I have
        # to deduplicate the computation
        ddt_oblique_ = r*(-s)*sint
        ddt_oblique = 2 * ddt_oblique_
    else:
        s = dy
        oblique_term = 2 * s * r * sint
        ddr_oblique = 2 * r + 2 * s * sint
        ddt_oblique_ = r*s*cost
        ddt_oblique = 2 * ddt_oblique_

    aggregate_term = r * r + oblique_term + s * s
    csq = c * c
    c3 = csq * c
    # d/dr first
    num = c * ddr_oblique
    phi_kernel = (1 + kappa) * csq * aggregate_term
    phi = np.sqrt(1 - phi_kernel)
    phip1 = 1 + phi
    phip1sq = phip1 * phip1
    den = phip1
    term1 = num / den

    num = c3 * (1+kappa)*ddr_oblique * aggregate_term
    den = (2 * phi) * phip1sq
    term2 = num / den
    dr = term1 + term2

    # d/dt
    num = c * ddt_oblique
    den = phip1
    term1 = num / den

    num = c3 * (1+kappa) * ddt_oblique_ * aggregate_term
    den = phi * phip1sq
    term2 = num / den
    dt = term1 + term2

    return dr, dt


def off_axis_conic_sigma(c, kappa, r, t, dx, dy=0):
    """Lowercase sigma (direction cosine projection term) for an off-axis conic.

    See Eq. (5.2) of oe-20-3-2483.

    Parameters
    ----------
    c : float
        axial curvature of the conic
    kappa : float
        conic constant
    r : numpy.ndarray
        radial coordinate, where r=0 is centered on the off-axis section
    t : numpy.ndarray
        azimuthal coordinate
    dx : float
        shift of the surface in x with respect to the base conic vertex,
        mutually exclusive to dy (only one may be nonzero)
        use dx=0 when dy != 0
    dy : float
        shift of the surface in y with respect to the base conic vertex

    Returns
    -------
    sigma(r,t)

    """
    if dy != 0 and dx != 0:
        raise ValueError('only one of dx/dy may be nonzero')

    if dx != 0:
        s = dx
        oblique_term = 2 * s * r * np.cos(t)
    else:
        s = dy
        oblique_term = 2 * s * r * np.sin(t)

    aggregate_term = r * r + oblique_term + s * s
    csq = c * c
    num = np.sqrt(1 - (1+kappa) * csq * aggregate_term)
    den = np.sqrt(1 - kappa * csq * aggregate_term)  # flipped sign, 1-kappa
    return num / den


def off_axis_conic_sigma_der(c, kappa, r, t, dx, dy=0):
    """Derivatives of 1/off_axis_conic_sigma.

    See Eq. (5.2) of oe-20-3-2483.

    Parameters
    ----------
    c : float
        axial curvature of the conic
    kappa : float
        conic constant
    r : numpy.ndarray
        radial coordinate, where r=0 is centered on the off-axis section
    t : numpy.ndarray
        azimuthal coordinate
    dx : float
        shift of the surface in x with respect to the base conic vertex,
        mutually exclusive to dy (only one may be nonzero)
        use dx=0 when dy != 0
    dy : float
        shift of the surface in y with respect to the base conic vertex

    Returns
    -------
    numpy.ndarray, numpy.ndarray
        d/dr(z), d/dt(z)

    """
    if dy != 0 and dx != 0:
        raise ValueError('only one of dx/dy may be nonzero')

    cost = np.cos(t)
    sint = np.sin(t)
    if dx != 0:
        s = dx
        oblique_term = 2 * s * r * cost
        ddr_oblique = 2 * r + 2 * s * cost
        # I accept the evil in writing this the way I have
        # to deduplicate the computation
        ddt_oblique_ = r*(-s)*sint
    else:
        s = dy
        oblique_term = 2 * s * r * sint
        ddr_oblique = 2 * r + 2 * s * sint
        ddt_oblique_ = r*s*cost

    aggregate_term = r * r + oblique_term + s * s
    csq = c * c
    # d/dr first
    phi_kernel = (1 + kappa) * csq * aggregate_term
    phi = np.sqrt(1 - phi_kernel)
    notquitephi_kernel = kappa * csq * aggregate_term
    notquitephi = np.sqrt(1 + notquitephi_kernel)

    num = csq * (1 + kappa) * ddr_oblique * notquitephi
    den = 2 * (1 - phi_kernel) ** (3/2)
    term1 = num / den

    num = csq * kappa * ddr_oblique
    den = 2 * phi * notquitephi
    term2 = num / den
    dr = term1 + term2

    # d/dt
    num = csq * (1+kappa) * ddt_oblique_ * notquitephi
    den = (1 - phi_kernel) ** (3/2)  # phi^3?
    term1 = num/den

    num = csq * kappa * ddt_oblique_
    den = phi * notquitephi
    term2 = num / den
    dt = term1 + term2  # minus in writing, but sine/cosine
    return dr, dt


def Q2d_and_der(cm0, ams, bms, x, y, normalization_radius, c, k, dx=0, dy=0):
    """Q-type freeform surface, with base (perhaps shifted) conicoic.

    Parameters
    ----------
    cm0 : iterable
        surface coefficients when m=0 (inside curly brace, top line, Eq. B.1)
        span n=0 .. len(cms)-1 and mus tbe fully dense
    ams : iterable of iterables
        ams[0] are the coefficients for the m=1 cosine terms,
        ams[1] for the m=2 cosines, and so on.  Same order n rules as cm0
    bms : iterable of iterables
        same as ams, but for the sine terms
        ams and bms must be the same length - that is, if an azimuthal order m
        is presnet in ams, it must be present in bms.  The azimuthal orders
        need not have equal radial expansions.

        For example, if ams extends to m=3, then bms must reach m=3
        but, if the ams for m=3 span n=0..5, it is OK for the bms to span n=0..3,
        or any other value, even just [0].
    x : numpy.ndarray
        X coordinates
    y : numpy.ndarray
        Y coordinates
    normalization_radius : float
        radius by which to normalize rho to produce u
    c : float
        curvature, reciprocal radius of curvature
    k : float
        kappa, conic constant
    rhosq : numpy.ndarray
        squared radial coordinate (non-normalized)
    dx : float
        shift of the base conic in x
    dy : float
        shift of the base conic in y

    Returns
    -------
    numpy.ndarray, numpy.ndarray, numpy.ndarray
        sag, dsag/drho, dsag/dtheta

    """
    # Q portion
    r, t = cart_to_polar(x, y)
    r2 = r / normalization_radius
    # content of curly braces in B.1 from oe-20-3-2483
    z, zprimer, zprimet = compute_z_zprime_Q2d(cm0, ams, bms, r2, t)

    base_sag = off_axis_conic_sag(c, k, r, t, dx, dy)
    base_primer, base_primet = off_axis_conic_der(c, k, r, t, dx, dy)

    # Eq. 5.1/5.2
    sigma = off_axis_conic_sigma(c, k, r, t, dx, dy)
    sigma = 1 / sigma
    sigmaprimer, sigmaprimet = off_axis_conic_sigma_der(c, k, r, t, dx, dy)

    zprimer /= normalization_radius
    zprimer2 = product_rule(sigma, z, sigmaprimer, zprimer)
    zprimet2 = product_rule(sigma, z, sigmaprimet, zprimet)
    z *= sigma
    z += base_sag
    zprimer2 += base_primer
    zprimet2 += base_primet
    return z, zprimer2, zprimet2


def _ensure_P_vec(P):
    if not hasattr(P, '__iter__'):
        P = np.array([0, 0, P], dtype=config.precision)
    else:
        # iterable
        P2 = np.zeros(3, dtype=config.precision)
        P2[-len(P):] = P
        P = P2

    return np.asarray(P).astype(config.precision)


def _none_or_rotmat(R):
    if R is None:
        return None
    if type(R) in (list, tuple):
        R = make_rotation_matrix(R)

    return R


def _map_stype(typ):
    if isinstance(typ, int):
        return typ

    typ = typ.lower()
    if typ in ('refl', 'reflect'):
        return STYPE_REFLECT

    if typ in ('refr', 'refract'):
        return STYPE_REFRACT

    if typ in ('eval'):
        return STYPE_EVAL


def _validate_n_and_typ(n, typ):
    if typ == STYPE_REFRACT and n is None:
        raise ValueError('refractive surfaces must have a refractive index function, not None')
    return


STYPE_REFLECT = -1
STYPE_REFRACT = -2
STYPE_EVAL =    -3  # NOQA


class Surface:
    """A surface for raytracing."""
    def __init__(self, typ, P, n, FFp, R=None, params=None, bounding=None):
        """Create a new surface for raytracing.

        Parameters
        ----------
        typ : int or str
            if an int, must be one of the STYPE constants
            if a str, must be something in the set {'refl', 'reflect', 'refr', 'refract', 'eval'}
            the type of surface (reflection, refraction, no ray bend)
        P : numpy.ndarray
            global surface position, [X,Y,Z]
        n : callable n(wvl) -> refractive index
            a function which returns the index of refraction at the given wavelength
        FFp : callable of signature F(x,y) -> z, [Nx, Ny]
            a function which returns the surface sag at point x, y as well as
            the X and Y partial derivatives at that point
        R : numpy.ndarray
            rotation matrix, may be None
        params : dict, optional
            surface type specific parameters
        bounding : dict, optional
            bounding geometry description
            at the moment, outer_radius and inner_radius are the only values
            which are used for anything.  More will be added in the future

        """
        typ = _map_stype(typ)
        P = _ensure_P_vec(P)
        R = _none_or_rotmat(R)
        _validate_n_and_typ(n, typ)

        self.typ = typ
        self.P = P
        self.n = n
        self.FFp = FFp
        self.R = R
        self.params = params
        self.bounding = bounding

    def sag_normal(self, x, y):
        """Sag z and normal [Fx, Fy, Fz] of the surface at the point (x,y).

        Parameters
        ----------
        x : numpy.ndarray
            x coordinate, non-normalized
        y : numpy.ndarray
            y coordinate, non-normalized

        Returns
        -------
        numpy.ndarray
            surface sag in Z

        """
        z, Fx, Fy = self.FFp(x, y)
        # faster than ones
        Fz = np.array([1.], dtype=config.precision)
        Fz = np.broadcast_to(Fz, Fx.shape)
        # F(X,Y,Z) = 0 = Z - F(x,Y)
        # d/dx, d/dy have leading - term, dz = 1 always
        der = np.stack([-Fx, -Fy, Fz], axis=1)
        return z, der

    @classmethod
    def conic(cls, c, k, typ, P, n=None, R=None, bounding=None):
        """Conic surface type.

        for documentation on typ, P, N, R, and bounding see the docstring for
        Surface.__init__

        Parameters
        ----------
        c : float
            vertex curvature
        k : float
            conic constant
            -1 = parabola
            0 = sphere
            < - 1 = hyperbola
            > - 1 = ellipse

        Returns
        -------
        Surface
            a conic surface

        """
        params = dict()
        params['c'] = c
        params['k'] = k

        def FFp(x, y):
            # TODO: significantly cheaper without t?
            r, t = cart_to_polar(x, y, vec_to_grid=False)
            rsq = r * r
            z = conic_sag(params['c'], params['k'], rsq)
            dr = conic_sag_der(params['c'], params['k'], r)
            dx, dy = surface_normal_from_cylindrical_derivatives(dr, 0, r, t)
            return z, dx, dy

        return cls(typ=typ, P=P, n=n, FFp=FFp, R=R, params=params, bounding=bounding)

    @classmethod
    def off_axis_conic(cls, c, k, typ, P, dy, dx=0, n=None, R=None, bounding=None):
        """Off-axis conic surface type.

        for documentation on typ, P, N, R, and bounding see the docstring for
        Surface.__init__

        Parameters
        ----------
        c : float
            vertex curvature
        k : float
            conic constant
            -1 = parabola
            0 = sphere
            < - 1 = hyperbola
            > - 1 = ellipse
        dy : float
            off-axis distance in y
        dx : float
            off-axis distance in x

        Returns
        -------
        Surface
            a conic surface

        """
        params = dict()
        params['c'] = c
        params['k'] = k
        params['dx'] = dx
        params['dy'] = dy

        def FFp(x, y):
            r, t = cart_to_polar(x, y, vec_to_grid=False)
            c, k, dx, dy = params['c'], params['k'], params['dx'], params['dy']
            z = off_axis_conic_sag(c, k, r, t, dx=dx, dy=dy)
            dr, dt = off_axis_conic_der(c, k, r, t, dx=dx, dy=dy)
            ddx, ddy = surface_normal_from_cylindrical_derivatives(dr, dt, r, t)
            return z, ddx, ddy

        return cls(typ=typ, P=P, n=n, FFp=FFp, R=R, params=params, bounding=bounding)

    @classmethod
    def plane(cls, typ, P, n=None, R=None, bounding=None):
        """A plane normal to its local Z axis.

        for documentation on typ, P, N, R, and bounding see the docstring for
        Surface.__init__

        The name of this will change in the future, likely to "plane."

        Returns
        -------
        Surface
            a stop

        """

        def FFp(x, y):
            zero = np.array([0.], dtype=x.dtype)
            zero_up = np.broadcast_to(zero, x.shape)
            return zero_up, zero_up, zero_up

        return cls(typ=typ, P=P, n=n, FFp=FFp, R=R, bounding=bounding)

    @classmethod
    def sphere(cls, c, typ, P, n, R=None, bounding=None):
        """A spherical surface.

        for documentation on typ, P, N, R, and bounding see the docstring for
        Surface.__init__

        Parameters
        ----------
        c : float
            vertex curvature

        Returns
        -------
        Surface
            a spherical surface

        """
        # TODO: cheaper implementation without conic
        return cls.conic(c=c, k=0, typ=typ, P=P, n=n, R=R, bounding=bounding)
