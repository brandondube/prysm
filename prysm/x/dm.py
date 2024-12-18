"""Deformable Mirrors."""
import copy

import numpy as truenp

from prysm.conf import config
from prysm.mathops import np, fft, is_odd
from prysm.fttools import forward_ft_unit, fourier_resample, crop_center, pad2d
from prysm.convolution import apply_transfer_functions
from prysm.coordinates import (
    warp,
    apply_homography,
    make_rotation_matrix,
    drop_z_3d_transformation,
    make_homomorphic_translation_matrix,
    promote_3d_transformation_to_homography,
)


def prepare_actuator_lattice(shape, Nact, sep, dtype):
    # Prepare a lattice of actuators.
    #
    # Usage guide:
    # returns a dict of
    # {
    #     mask, shape Nact
    #     actuators, shape Nact
    #     poke_arr, shape shape
    #     ixx, shape (truthy part of mask)
    #     iyy, shape (truthy part of mask)
    # }
    #
    # assign poke_arr[iyy, ixx] = actuators[mask] in the next step
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

    ix = slice(neg_extreme_x, pos_extreme_x, skip_samples_x)
    iy = slice(neg_extreme_y, pos_extreme_y, skip_samples_y)
    ixx = ix
    iyy = iy

    poke_arr = np.zeros(shape, dtype=dtype)
    return {
        'actuators': actuators,
        'poke_arr': poke_arr,
        'ixx': ixx,
        'iyy': iyy,
    }


def prepare_fwd_reverse_projection_coordinates(shape, rot):
    # 1. make the matrix that describes the rigid body transformation
    # 2. make the coordinate grid (in "pixels") for the data
    # 3. project the coordinates "forward" (for forward_model())
    # 4. project the coordinates "backwards" (for backprop)
    R = make_rotation_matrix(rot)
    oy, ox = [(s-1)/2 for s in shape]
    y, x = [np.arange(s, dtype=config.precision) for s in shape]
    x, y = np.meshgrid(x, y)
    Tin = make_homomorphic_translation_matrix(-ox, -oy)
    Tout = make_homomorphic_translation_matrix(ox, oy)
    R = promote_3d_transformation_to_homography(R)
    Mfwd = Tout@(R@Tin)
    Mfwd = drop_z_3d_transformation(Mfwd)
    Mifwd = np.linalg.inv(Mfwd)
    xfwd, yfwd = apply_homography(Mifwd, x, y)
    xrev, yrev = apply_homography(Mfwd, x, y)
    return (xfwd, yfwd), (xrev, yrev)


class DM:
    """A DM whose actuators fill a rectangular region on a perfect grid, and have the same influence function."""
    def __init__(self, ifn, Nout, Nact=50, sep=10, shift=(0, 0), rot=(0, 0, 0),
                 upsample=1, project_centering='fft'):
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
        ifn : ndarray
            influence function; assumes the same for all actuators and must
            be the same shape as (x,y).  Assumed centered on N//2th sample of x, y.
            Assumed to be well-conditioned for use in convolution, i.e.
            compact compared to the array holding it
        Nout : int or tuple of int, length 2
            number of samples in the output array; see notes for details
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
        project_centering : str, {'fft', 'interpixel'}
            how to deal with centering when projecting the surface into the beam normal
            fft = the N/2 th sample, rounded to the right, defines the origin.
            interpixel = the N/2 th sample, without rounding, defines the origin

        Notes
        -----
        If ifn is 500x500 and upsample=0.5, then the nominal output array is
        250x250.  If this is supposed to line up with a pupil embedded in a
        512x512 array, then the user would have to call pad2d after, which is
        slightly worse than one stop shop.

        The Nout parameter allows the user to specify Nout=512, and the DM's
        render method will internally do the zero-pad or crop necessary to
        achieve the desired array size.

        """
        if isinstance(Nout, int):
            Nout = (Nout, Nout)
        if isinstance(Nact, int):
            Nact = (Nact, Nact)
        if isinstance(sep, int):
            sep = (sep, sep)

        s = ifn.shape

        # stash inputs and some computed values on self
        self.ifn = ifn
        self.Ifn = fft.fft2(ifn)
        self.Nout = Nout
        self.Nact = Nact
        self.sep = sep
        self.shift = shift
        self.obliquity = truenp.cos(truenp.radians(truenp.linalg.norm(rot)))
        self.rot = rot
        self.upsample = upsample

        # prepare the poke array and supplimentary integer arrays needed to
        # copy it into the working array
        out = prepare_actuator_lattice(ifn.shape, Nact, sep, dtype=ifn.dtype)
        self.actuators = out['actuators']
        self.actuators_work = np.zeros_like(self.actuators)
        self.poke_arr = out['poke_arr']
        self.ixx = out['ixx']
        self.iyy = out['iyy']

        self.needs_rot = True
        if np.allclose(rot, [0, 0, 0]):
            self.needs_rot = False
            self.projx = None
            self.projy = None
            self.invprojx = None
            self.invprojy = None
        else:
            fwd, rev = prepare_fwd_reverse_projection_coordinates(s, rot)
            self.projx, self.projy = fwd
            self.invprojx, self.invprojy = rev

        # shift data
        if shift[0] != 0 or shift[1] != 0:
            # caps = Fourier variable (x -> X, y -> Y)
            # make 2pi/px phase ramps in 1D (much faster)
            # then broadcast them to 2D when they're used as transfer functions
            # in a Fourier convolution
            Y, X = [forward_ft_unit(1, s, shift=False) for s in s]
            Xramp = np.exp(X * (-2j * np.pi * shift[0]))
            Yramp = np.exp(Y * (-2j * np.pi * shift[1]))
            shpx = s
            shpy = tuple(reversed(s))
            Xramp = np.broadcast_to(Xramp, shpx)
            Yramp = np.broadcast_to(Yramp, shpy).T
            self.Xramp = Xramp
            self.Yramp = Yramp
            self.tf = [self.Ifn * self.Xramp * self.Yramp]
        else:
            self.tf = [self.Ifn]

    def copy(self):
        """Make a (deep) copy of this DM."""
        return copy.deepcopy(self)

    def update(self, actuators):
        # semantics for update:
        # the mask is non-none, then actuators is a 1D vector of the same size
        # as the nonzero elements of the mask
        #
        # or mask is None, and actuators is 2D
        self.actuators[:] = actuators[:]
        return

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
        ndarray
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
        self.poke_arr[self.iyy, self.ixx] = self.actuators

        # self.dx is unused inside apply tf, but :shrug:
        sfe = apply_transfer_functions(self.poke_arr, None, self.tf, shift=False)
        if self.needs_rot:
            warped = warp(sfe, self.projx, self.projy)
        else:
            warped = sfe
        if wfe:
            warped *= (2*self.obliquity)

        if self.upsample != 1:
            warped = fourier_resample(warped, self.upsample)

        self.Nintermediate = warped.shape

        if warped.shape[0] < self.Nout[0]:
            # need to pad
            warped = pad2d(warped, out_shape=self.Nout)
        elif warped.shape[0] > self.Nout[1]:
            warped = crop_center(warped, out_shape=self.Nout)

        return warped

    def render_backprop(self, protograd, wfe=True):
        """Gradient backpropagation for render().

        Parameters
        ----------
        protograd : ndarray
            "prototype gradient"
            the array holding the work-in-progress towards the gradient.
            For example, in a problem fitting actuator commands to a surface,
            you might have:

            render() returns a 512x512 array, for 48x48 actuators.
            y contains a 512x512 array of target surface heights

            The euclidean distance between the two as a cost function:
            cost = np.sum(abs(render() - y)**2)

            Then the first step in computing the gradient is
            diff = 2 * (render() - y)

            and you would call
            dm.render_backprop(diff)
        wfe : bool, optional
            if True, the return is scaled as for a wavefront error instead
            of surface figure error

        Returns
        -------
        ndarray
            analytic gradient, shape Nact x Nact

        Notes
        -----
        Not compatible with complex valued protograd

        """
        """Gradient backpropagation for self.render."""
        if protograd.shape[0] > self.Nintermediate[0]:
            # forward padded, we need to crop
            protograd = crop_center(protograd, out_shape=self.Nintermediate)
        elif protograd.shape[0] < self.Nintermediate[0]:
            # forward cropped, we need to pad
            protograd = pad2d(protograd, out_shape=self.Nintermediate)

        if self.upsample != 1:
            upsample = self.ifn.shape[0]/protograd.shape[0]
            protograd = fourier_resample(protograd, upsample)

        if wfe:
            protograd *= (2*self.obliquity)

        # return protograd
        if self.needs_rot:
            protograd = warp(protograd, self.invprojx, self.invprojy)

        # return protograd
        in_actuator_space = apply_transfer_functions(protograd, None, np.conj(self.tf), shift=False)
        return in_actuator_space[self.iyy, self.ixx]
