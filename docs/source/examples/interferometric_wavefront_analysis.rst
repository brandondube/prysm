##########################################
Analysis of Interferometric Wavefront Data
##########################################

In this example, we will see how to use prysm to almost entirely supplant the software that comes with a commerical interferometer to analyze the wavefront of an optic.  We begin by importing the relevant classes and setting some aesthetics for matplotlib.

>>> from pathlib import Path
>>>
>>> from matplotlib import pyplot as plt
>>>
>>> from prysm import Interferogram, FringeZernike
>>> from prysm.fringezernike import fit
>>>
>>> plt.style.use('bmh')

We point prysm to the file, create a new interferogram, mask it to a circular region 100 mm across, subtract piston, tip/tilt and power, and evalute the PV and RMS wavefront error.  We also plot the wavefront to make sure all has gone well

>>> p = Path('~').expanduser() / 'Desktop' / 'sample_file.dat'
>>> i = Interferogram.from_zygo_dat(p)
>>> i.crop().mask('circle', 100).crop()
>>> i.remove_piston_tiptilt_power()
>>> print(i.pv, i.rms)
>>> i.plot2d(clim=100)  # +/- 100 nm

The interferogram is cropped twice -- once to enclose the valid data, then again to apply a mask centered on that region.  For relatively conventional interferometry, you may want to stop here.  If you want to use a different unit, that is easy enough,

>>> i.change_phase_unit('waves')
>>> print(i.pv, i.rms)

Perhaps your part is a mirror with a central obscuration,

>>> # do the first cicle too -- already done here.
>>> i.mask('invertedcircle', 25)

There is no need to crop again since the outer bound has not changed.  Perhaps you wish to evaluated the RMS within the 1 - 10 mm spatial periods,

>>> i.change_phase_unit('nm')  # nm, please.
>>> i.fill()
>>> print(i.bandlimited_rms(1,10))

This value is derived from the PSD, so you must call fill first.  Do not worry about the corners of the array containing data - it will be windowed out.  If you do this on a part which has a central obscuration, the result will be incorrect.  Otherwise, the value tends to agree to within +/- 5% of Zygo's Mx ® software, though the authors of prysm do not believe they are calculated at all the same way.

If you wish to decompose the wavefront into Zernike polynomials, that is easy enough.

>>> # do this on data which has not been filled
>>> coefficients = fit(i.phase, terms=36, norm=True)
>>> fz = FringeZernike(coefficients, dia=i.diameter, opd_unit=i.phase_unit, norm=True)
>>> print(fz)

This print might be a bit daunting, one may prefer to see the top few terms by magnitude,

>>> print(fz.top_n(5))

or a barplot of all terms,

>>> fz.barplot_magnitudes(orientation='v')
