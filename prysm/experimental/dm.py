"""Deformable Mirrors."""

from prysm.mathops import np
from prysm.convolution import conv
from prysm.coordinates import (
    make_rotation_matrix,
    apply_rotation_matrix,
    xyXY_to_pixels,
    regularize,

)


def prepare_actuator_lattice(shape, dx, Nact, sep, shift, mask, dtype):
    """Prepare a lattice of actuators.

    Usage guide:
    returns a dict of
    {
        mask; shape Nact
        actuators; shape Nact
        poke_arr; shape shape
        ixx; shape (truthy part of mask)
        iyy; shape (truthy part of mask)
    }

    assign poke_arr[iyy, ixx] = actuators[mask] in the next step
    """
    if mask is None:
        mask = np.ones(Nact, dtype=bool)

    actuators = np.ones(Nact, dtype=dtype)

    cy, cx = [s//2 for s in shape]
    Nactx, Nacty = Nact
    skip_samples_x, skip_samples_y = [int(s/dx) for s in sep]
    # python trick; floor division (//) rounds to negative inf, not zero
    # because FFT grid alignment biases things to the left, if Nact is odd
    # we want more on the negative side;
    # this will make that so
    neg_extreme_x = cx + -Nactx//2 * skip_samples_x
    neg_extreme_y = cy + -Nacty//2 * skip_samples_y
    pos_extreme_x = cx + Nactx//2 * skip_samples_x
    pos_extreme_y = cy + Nacty//2 * skip_samples_y

    # ix = np.arange(neg_extreme_x, pos_extreme_x+skip_samples_x, skip_samples_x)
    # iy = np.arange(neg_extreme_y, pos_extreme_y+skip_samples_y, skip_samples_y)
    # ixx, iyy = np.meshgrid(ix, iy)
    # ixx = ixx[mask]
    # iyy = iyy[mask]
    ix = slice(neg_extreme_x, pos_extreme_x, skip_samples_x)
    iy = slice(neg_extreme_y, pos_extreme_y, skip_samples_y)
    ixx = ix
    iyy = iy

    poke_arr = np.zeros(shape, dtype=dtype)
    return {
        'mask': mask,
        'actuators': actuators,
        'poke_arr': poke_arr,
        'ixx': ixx,
        'iyy': iyy,
    }


class SimpleDM:
    """A DM whose actuators fill a rectangular region on a perfect grid, and have the same influence function."""
    def __init__(self, x, y, ifn, Nact=(50, 50), sep=(0.4, 0.4), shift=(0, 0), rot=(0, 10, 0), mask=None):
        """Create a new DM model.

        Parameters
        ----------
        x : numpy.ndarray
            x coordinates at the DM surface; 2D
        y : numpy.ndarray
            y coordinates at the DM surface; 2D
        ifn : numpy.ndarray
            influence function; assumes the same for all actuators and must
            be the same shape as (x,y).  Assumed centered on N//2th sample of x,y.
        Nact : tuple of int, length 2
            (X, Y) actuator counts
        sep : tuple of int, length 2
            (X, Y) actuator separation / pitch
        shift : tuple of int, length 2
            (X, Y) shift of the actuator grid to the N//2th sample of x and y
            (~= 0, assumes FFT grid alignment)
        rot : tuple of int, length <= 3
            (Z, Y, X) rotations; see coordinates.make_rotation_matrix
        mask : numpy.ndarray
            boolean ndarray of shape Nact used to suppress/delete/exclude
            actuators; 1=keep, 0=suppress

        """
        dx = x[0, 1] - x[0, 0]

        self.x = x
        self.y = y
        self.ifn = ifn
        self.Nact = Nact
        self.sep = sep
        self.shift = shift
        self.dx = dx
        self.obliquity = np.cos(np.radians(np.linalg.norm(rot)))
        self.rot = rot

        out = prepare_actuator_lattice(x.shape, dx, Nact, sep, shift, mask, dtype=x.dtype)
        self.mask = out['mask']
        self.actuators = out['actuators']
        self.actuators_work = np.zeros_like(self.actuators)
        self.poke_arr = out['poke_arr']
        self.ixx = out['ixx']
        self.iyy = out['iyy']

        rotmat = make_rotation_matrix(rot)
        XY = apply_rotation_matrix(rotmat, x, y)
        XY2 = xyXY_to_pixels((x, y), XY)
        self.XY = XY
        self.XY2 = XY2

    def render(self, wfe=True):
        """Render the DM's surface figure or wavefront error.

        Parameters
        ----------
        wfe : bool, optional
            if True, converts the "native" surface figure error into
            reflected wavefront error, by multiplying by 2 times the obliquity.
            obliquity is the cosine of the rotation vector.

        Returns
        -------
        numpy.ndarray
            surface figure error or wfe, projected into the beam normal
            by self.rot

        """
        # optimization (or not):
        # actuators is small, say 40x40
        # while poke_arr is ~= 10x the resolution (400x400)
        #
        # it is most optimal to set the values of poke_arr based on the mask
        # however, for such small arrays it makes little difference and the
        # code appears much less expressive
        # what is here is ~99.1% of the speed with better legibility

        # potential "bug" - it is assumed the content of actuators_work
        # where the actuators are masked off is zero, or whatever the desired
        # sticking value is.  If the expected behavior for masked actuators
        # changes over the life of this instance, the user may be surprised
        # OTOH, it may be a "feature" that stuck actuators, etc, may be
        # adjusted in this way rather elegantly
        self.actuators_work[self.mask] = self.actuators[self.mask]
        self.poke_arr[self.iyy, self.ixx] = self.actuators_work

        # technically the args are in the wrong order here
        sfe = conv(self.poke_arr, self.ifn)
        warped = regularize(xy=None, XY=self.XY, z=sfe, XY2=self.XY2)
        if wfe:
            warped *= (2*self.obliquity)

        return warped
