******
Pupils
******

Most any physical optics model begins with a description of a wave at a pupil plane.  This page will cover the core functionality of pupils; each analytical variety has its own documentation.


All `Pupil` parameters have default values, so one may be created with no arguments;

>>> from prysm import Pupil
>>> p = Pupil()

Pupils will be modeled using square arrays, the shape of which is controlled by a *samples* argument.  They also accept an *epd* argument which controls their diameter in mm, as well as a *wavelength* which sets the wavelength of light used in microns.  There is also an *opd_unit* argument that tells prysm what units are used to describe the *phase* associated with the pupil.  Finally, a *mask* may be specified, either as a string using prysm's built-in masking capabilities, or as an array of the same shape as the pupil.  Putting it all together,

>>> p = Pupil(samples=123, epd=456.7, wavelength=1.0, opd_unit='nm', mask='dodecagon')

`p` is a pupil with a 12-sided aperture backed by a 123x123 array which spans 456.7 mm and is impinged on by light of wavelength 1 micron.

Pupils have some more advanced parameters.  *mask_target* determines if the phase ('phase'), wavefunction ('fcn'), or both ('both') will be masked.  When embedding prysm in a task that repeatedly creates pupils, e.g. an optimizer for wavefront sensing, applying the mask to the phase is wasted computation and can be avoided.

If you wish to provide your own data for a pupil model, simply provide the *ux*, *uy*, and *phase* arguments, which are the x and y unit axes of shape (n,) and (m,), and *phase* is in units of opd_unit and of shape (m,n).

>>> p = Pupil(ux=x, uy=y, phase=phase, opd_unit='um')

Once a pupil is created, you can access quick access slices,

>>> p.slice_x  # returns tuple of unit, slice_of_phase
>>> p.slice_y

or evaluate the wavefront,

>>> p.pv, p.rms  # in units of opd_unit

The pupil may also be plotted.  Plotting functions have defaults for all arguments, but may be overriden

>>> p.plot2d(cmap='RdYlBu', clim=(-100,100), interp_method='sinc')

*cmap*, *clim* and *interp_method* are passed directly to matplotlib.  A figure and axis may also be provided if you would like to control these, for e.g. making a figure with multiple axes for different stages of the model

>>> from matplotlib import pyplot as plt
>>> fig, ax = plt.subplots(figsize=(10,10))
>>> p.plot2d(fig=fig, ax=ax)

A synthetic interferogram may be generated,

>>> fig, ax = plt.subplots(figsize=(4,4))
>>> p.interferogram(passes=2, visibility=0.5, fig=fig, ax=ax)

Pupils also support addition and subtraction,

>>> p2 = p + p - p

these actions modify the *phase* attribute of a pupil out-of-place.  After doing so, you must explicitly update the wavefunction,

>>> p2._phase_to_wavefunction()

or downstream objects that rely on optical propagation e.g. PSFs will be incorrect.

The complete API documentation is below.

----

.. automodule:: prysm.pupil
    :members:
    :undoc-members:
    :show-inheritance:
