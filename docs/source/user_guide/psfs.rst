****
PSFs
****

PSFs in prysm have a very simple constructor;

>>> import numpy as np
>>> from prysm import PSF
>>> x = y = np.linspace(-1,1,128)
>>> z = np.random.random((128,128))
>>> ps = PSF(data=z, unit_x=x, unit_y=y)

The encircled energy can be computed, for either a single point or an iterable (tuple, list, numpy array, ...) of points;

>>> print(ps.encircled_energy(0.1), ps.encircled_energy([0.1, 0.2, 0.3])
12.309576159990891, array([12.30957616, 24.61581586, 36.92244558])

encircled energy is computed via the method described in  V Baliga, B D Cohn, *"Simplified Method For Calculating Encircled Energy,"* Proc. SPIE 0892, *Simulation and Modeling of Optical Systems*, (9 June 1988).

The inverse can also be computed using the L-BFGS-B nonlinear optimization routine, but the wavelength and F/# must be provided for the initial guess,

>>> ps.wavelength, ps.fno = 0.5, 2
>>> print(ps.ee_radius(1))
0.008104694339936169

Baliga's method is relatively slow for large arrays, so a dictionary is kept of all computed encircled energies at :code:`ps._ee`.  The encircled energy can be plotted.  An axis limit must be provided if no encircled energy values have been computed.  If some have, by default prysm will plot the computed values if no axis limit is given

>>> ps.plot_encircled_energy()

or

>>> ps.plot_encircled_energy(axlim=1, npts=50)

The PSF can be plotted in 2D,

>>> # ~0.838, exact value of energy contained in first airy zero
>>> from prysm.psf import FIRST_AIRY_ENCIRCLED

>>> ps.plot2d(axlim=0.8, power=2, interp_method='sinc', pix_grid=0.2, show_axlabels=True, show_colorbar=True, circle_ee=FIRST_AIRY_ENCIRCLED)

Both :code:`plot_encircled_energy` and :code:`plot2d` take the usual :code:`fig` and :code:`ax` kwargs as well.  For plot2d, the axlim arg sets the x and y axis limits to symmetrical values of :code:`axlim`, i.e. the limits above will be [0.8, 0.8], [0.8, 0.8].  :code:`power` controls the stretch of the color scale.  The image will be stretched by the 1/power power, e.g. 2 plots psf^(1/2).  :code:`interp_method` is passed to matplotlib.  :code:`pix_grid` will use the minor axis ticks to draw a light grid over the PSF, intended to show the size of a PSF relative to the pixels of a detector.  Units of microns.  :code:`show_axlabels` and :code:`show_colorbar` both default to True, and control whether the axis labels are set and if the colorbar is drawn.  :code:`circle_ee` will draw a dashed circle at the radius containing the specified portion of the energy, and another at the diffraction limited radius for a circular aperture.

PSFs are a subclass of :class:`Convolvable` and inherit all methods and attributes

.. autoclass:: prysm.psf.PSF
    :members:
    :undoc-members:
    :show-inheritance:
