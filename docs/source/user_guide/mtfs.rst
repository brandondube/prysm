****
MTFs
****


Prysm models often include analysis of Modulation Transfer Function (MTF) data.  The MTF is formally defined

    the normalized magnitude of the Fourier transform of the point spread function

It is nothing more and nothing less.  It may not be negative, complex-valued, or equal to any value other than unity at the origin.

Initializing an MTF model should feel similar to a PSF,

>>> import numpy as np
>>> from prysm import MTF
>>> x = y = 1/np.linspace(-1,1,128)
>>> z = np.random.random((128,128))
>>> mt = MTF(data=z, unit_x=x, unit_y=y)

MTFs are usually created from a PSF instance

>>> mt = MTF.from_psf(your_psf_instance)

If modeling the MTF directly from a pupil plane, the intermediate PSF plane may be skipped;

>>> mt = MTF.from_pupil(your_pupil_instance, Q=2, efl=2)

Much like a PSF or other Convolvable, MTFs have quick-access slices

>>> print(mt.tan)
    (array([...]), array([...]))

>>> print(mt.sag)
    (array([...]), array([...]))

The tangential MTF is a slice through the x=0 axis, and assumes the usual optics sign convention of an object extended in y.  The sagittal MTF is a slice through the y=0 axis.

The MTF at exact frequencies may be queried through any of the following methods:  :code:`(MTF).exact_polar`, takes arguments of freqs and azimuths.  If there is a single frequency and multiple azimuths, the MTF at each azimuth and and the specified radial spatial frequency will be returned.  The reverse is true for a single azimuth and multiple frequencies.  :code:`(MTF).exact_xy` follows the same semantics, but with Cartesian coordinates instead of polar.  :code:`(MTF).exact_tan` and :code:`(MTF).exact_sag` both take a single argument of freq, which may be an int, float, or ndarray.

Finally, MTFs may be plotted:


>>> mt.plot_tan_sag(max_freq=200, fig=None, ax=None, labels=('Tangential', 'Sagittal'))
>>> mt.plot2d(max_freq=200, power=1, fig=None, ax=None)

all arguments have these default values.  The axes of plot2d will span (-max_freq, max_freq) on both x and y.

The complete API documentation is below.

----

.. autoclass:: prysm.otf.MTF
    :members:
    :undoc-members:
    :show-inheritance:
