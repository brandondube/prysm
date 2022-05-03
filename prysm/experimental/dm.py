"""Deformable Mirrors."""

import warnings

import numpy as truenp

from prysm.mathops import np, fft, is_odd
from prysm.fttools import forward_ft_unit, fourier_resample
from prysm.convolution import apply_transfer_functions
from prysm.coordinates import (
    make_xy_grid,
    make_rotation_matrix,
    apply_rotation_matrix,
    xyXY_to_pixels,
    regularize,
)


def prepare_actuator_lattice(shape, Nact, sep, mask, dtype):
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

    actuators = np.zeros(Nact, dtype=dtype)

    cy, cx = [s//2 for s in shape]
    Nactx, Nacty = Nact
    skip_samples_x, skip_samples_y = sep
    # python trick; floor division (//) rounds to negative inf, not zero
    # because FFT grid alignment biases things to the left, if Nact is odd
    # we want more on the negative side;
    # this will make that so
    offx = 0
    offy = 0
    if not is_odd(Nactx):
        offx = skip_samples_x // 2
    if not is_odd(Nacty):
        offy = skip_samples_y // 2

    neg_extreme_x = cx + -Nactx//2 * skip_samples_x + offx
    neg_extreme_y = cy + -Nacty//2 * skip_samples_y + offy
    pos_extreme_x = cx + Nactx//2 * skip_samples_x + offx
    pos_extreme_y = cy + Nacty//2 * skip_samples_y + offy

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


class DM:
    """A DM whose actuators fill a rectangular region on a perfect grid, and have the same influence function."""
    def __init__(self, ifn, Nact=50, sep=10, shift=(0, 0), rot=(0, 0, 0), upsample=1, mask=None, project_centering='fft'):
        """Create a new DM model.

        This model is based on convolution of a 'poke lattice' with the influence
        function.  It has the following idiosyncracies:

            1.  The poke lattice is always "FFT centered" on the array, i.e.
                centered on the sample which would contain the DC frequency bin
                after an FFT.
            2.  The rotation is applied in the same sampling as ifn
            3.  Shifts and resizing are applied using a Fourier method and not
                subject to quantization

        Parameters
        ----------
        ifn : numpy.ndarray
            influence function; assumes the same for all actuators and must
            be the same shape as (x,y).  Assumed centered on N//2th sample of x, y.
            Assumed to be well-conditioned for use in convolution, i.e.
            compact compared to the array holding it
        Nact : int or tuple of int, length 2
            (X, Y) actuator counts
        sep : int or tuple of int, length 2
            (X, Y) actuator separation, samples of influence function
        shift : tuple of float, length 2
            (X, Y) shift of the actuator grid to (x, y), units of x influence
            function sampling.  E.g., influence function on 0.1 mm grid, shift=1
            = 0.1 mm shift.  Positive numbers describe (rightward, downward)
            shifts in image coordinates (origin lower left).
        rot : tuple of int, length <= 3
            (Z, Y, X) rotations; see coordinates.make_rotation_matrix
        upsample : float
            upsampling factor used in determining output resolution, if it is different
            to the resolution of ifn.
        mask : numpy.ndarray
            boolean ndarray of shape Nact used to suppress/delete/exclude
            actuators; 1=keep, 0=suppress
        project_centering : str, {'fft', 'interpixel'}
            how to deal with centering when projecting the surface into the beam normal
            fft = the N/2 th sample, rounded to the right, defines the origin.
            interpixel = the N/2 th sample, without rounding, defines the origin

        """
        if isinstance(Nact, int):
            Nact = (Nact, Nact)
        if isinstance(sep, int):
            sep = (sep, sep)

        s = ifn.shape
        self.x, self.y = make_xy_grid(s, dx=1)
        if project_centering.lower() == 'interpixel' and not is_odd(s[1]):
            self.x += 0.5
        if project_centering.lower() == 'interpixel' and not is_odd(s[0]):
            self.y += 0.5

        # stash inputs and some computed values on self
        self.ifn = ifn
        self.Ifn = fft.fft2(ifn)
        self.Nact = Nact
        self.sep = sep
        self.shift = shift
        self.obliquity = truenp.cos(truenp.radians(truenp.linalg.norm(rot)))
        self.rot = rot
        self.upsample = upsample

        # prepare the poke array and supplimentary integer arrays needed to
        # copy it into the working array
        out = prepare_actuator_lattice(ifn.shape, Nact, sep, mask, dtype=self.x.dtype)
        self.mask = out['mask']
        self.actuators = out['actuators']
        self.actuators_work = np.zeros_like(self.actuators)
        self.poke_arr = out['poke_arr']
        self.ixx = out['ixx']
        self.iyy = out['iyy']

        # rotation data
        self.rotmat = make_rotation_matrix(rot)
        XY = apply_rotation_matrix(self.rotmat, self.x, self.y)
        XY2 = xyXY_to_pixels(XY, (self.x, self.y))
        self.XY = XY
        self.XY2 = XY2
        self.needs_rot = True
        if np.allclose(rot, [0, 0, 0]):
            self.needs_rot = False

        # shift data
        if shift[0] != 0 or shift[1] != 0:
            # caps = Fourier variable (x -> X, y -> Y)
            # make 2pi/px phase ramps in 1D (much faster)
            # then broadcast them to 2D when they're used as transfer functions
            # in a Fourier convolution
            Y, X = [forward_ft_unit(1, s, shift=False) for s in self.x.shape]
            Xramp = np.exp(X * (-2j * np.pi * shift[0]))
            Yramp = np.exp(Y * (-2j * np.pi * shift[1]))
            shpx = self.x.shape
            shpy = tuple(reversed(self.x.shape))
            Xramp = np.broadcast_to(Xramp, shpx)
            Yramp = np.broadcast_to(Yramp, shpy).T
            self.Xramp = Xramp
            self.Yramp = Yramp
            self.tf = [self.Ifn * self.Xramp * self.Yramp]
        else:
            self.tf = [self.Ifn]

    def render(self, wfe=True, out=None):
        """Render the DM's surface figure or wavefront error.

        Parameters
        ----------
        wfe : bool, optional
            if True, converts the "native" surface figure error into
            reflected wavefront error, by multiplying by 2 times the obliquity.
            obliquity is the cosine of the rotation vector.
        out : numpy.ndarray
            output array to place the output in,
            if None, a new output array is allocated.
            If not None and self.upsample == 1, an extra copy will be performed
            and a warning emitted

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

        # self.dx is unused inside apply tf, but :shrug:
        sfe = apply_transfer_functions(self.poke_arr, None, self.tf, shift=False)
        if self.needs_rot:
            warped = regularize(xy=None, XY=self.XY, z=sfe, XY2=self.XY2)
        else:
            warped = sfe
        if wfe:
            warped *= (2*self.obliquity)

        if self.upsample != 1:
            warped = fourier_resample(warped, self.upsample)
        else:
            if out is not None:
                warnings.warn('prysm/DM: out was not None when upsample=1.  A wasteful extra copy was performed which reduces performance.')
                out[:] = warped[:]  # copy all elements
                warped = out
        return warped
