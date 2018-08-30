**************
Interferograms
**************

Prysm offers rich features for analysis of interferometric data.  :code:`Interferogram` objects are conceptually similar to :doc:`Pupils <./pupils>` and both inherit from the same base class, as they both have to do with optical phase.  The construction of an Interferogram requires only a few parameters:

>>> import numpy as np
>>> from prysm import Interferogram
>>> x = y = np.arange(129)
>>> z = np.random.uniform((128,128))
>>> interf = Interferogram(phase=z, intensity=None, unit_x=x, unit_y=y, scale='mm', phase_unit='nm', meta={'wavelength': 632.8e-9})

Notable are the scale, and phase unit, which define the xy and z units, respectively.  Any SI unit is accepted as well as angstroms.  Imperial units not accepted.  :code:`meta` is a dictionary to store metadata.  For several interferogram methods to work, the wavelength must be present *in meters.*  This departure from prysm's convention of preferred units is used to maintain compatability with Zygo dat files.  Interferograms are usually created from such files:

>>> interf = Interferogram.from_zygo_dat(your_path_file_object_or_bytes)

and both the dat and datx format from Zygo are supported.  Dat carries no dependencies, while datx requries the installation of h5py.  In addition to properties inherited from the OpticalPhase class (pv, rms, Sa), :code:`Interferograms` have a :code:`dropout_percentage` property, which gives the percentage of NaN values within the phase array.  These NaNs may be filled,

>>> interf.fill(_with=0)

with 0 as a default value; only constants are supported.  The modification is done in-place and the method returns :code:`self`.  Piston, tip-tilt, and power may be removed:

>>> interf.fill()\
>>>     .remove_piston()\
>>>     .remove_tiptilt()\
>>>     .remove_power()

again done in-place and returning self, so methods can be chained.  One line convenience wrappers exist:

>>> interf.remove_piston_tiptilt()
>>> interf.remove_piston_tiptilt_power()

Masks may be applied;

>>> your_mask = np.ones(interf.phase.shape)
>>> interf.mask(your_mask)

the phase is deleted (replaced with NaN) wherever the mask is equal to zero.  The same applies to the intensity.

Interferograms may be cropped, deleting empty (NaN) regions around a measurment;

>>> interf.crop()

The two dimensional Power Spectral Density (PSD) may be computed.  The data may not contain NaNs, and piston tip and tilt should be removed prior.  A 2D welch window is used, so there is no need for concern about zero values creating a discontinuity at the edge of circular or other nonrectangular apertures.

>>> interf.crop().remove_piston_tiptilt_power().fill()
>>> ux, uy, psd = interf.psd()

x, y, and azimuthally averaged cuts of the PSD are also available

>>> psd_dict = interf.psd_xy_avg()
>>> ux, psd_x = psd_dict['x']
>>> uy, psd_y = psd_dict['y']
>>> ur, psd_r = psd_Dict['avg']

The PSD computation is cached and the cache must be cleared if the data is modified and PSD recomputed;

>>> interf.clear_psd()

band-rejection filters may be applied,

>>> interf.bandreject(wllow=1, wlhigh=10)

and the PSD may be plotted,

>>> interf.plot_psd2d(axlim=1, interp_method='lanczos', fig=None, ax=None)
>>> interf.plot_psd_xyavg(xlim=(1e0,1e3), ylim=(1e-7,1e2), fig=None, ax=None)

For the x/y and average plot, a Lorentzian model may be plotted alongside the data for e.g. visual verification of a requirement:

>>> interf.plot_psd_xyavg(a=1,b=1,c=1)

The complete API documentation is below.

----

.. autoclass:: prysm.interferogram.Interferogram
    :members:
    :undoc-members:
    :show-inheritance:
