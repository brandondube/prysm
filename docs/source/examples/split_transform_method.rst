######################
Split Transform Method
######################

In this example we will demonstrate the split transform method for analyzing the imaging performance of a mirror.  We begin as usual by importing the relevant classes.

>>> from pathlib import Path
>>>
>>> from prysm import Interferogram, Pupil, PSF
>>>
>>> from matplotlib import pyplot as plt
>>> plt.style.use('bmh')

First, we load the interferogram and preprocess the data.

>>> p = Path('~').expanduser() / 'Desktop' / 'sample_file.dat'
>>> i = Interferogram.from_zygo_dat(p)
>>> i.crop().mask('circle', 100).crop()
>>> i.remove_piston_tiptilt_power()
>>> i.plot2d(clim=100)  # verify the phase looks OK

If you are dissatisfied with the masking prowess of prysm, it is recommended to use the software that came with your interferometer.  The order of operations from here is very important.  Because prysm modifies these classes in-place, we should propagate the PSF before filling the interferogram for PSF analysis.

>>> pu = Pupil.from_interferogram(i)
>>> psf = PSF.from_pupil(pu, efl=200, Q=2)  # F/2
>>> i.fill()

We can then plot,

>>> fig, ax = plt.subplots()
>>> i.plot_psd_xyavg(fig=fig, ax=ax)
>>> fig, ax = plt.subplots()
>>> # bicubic is highly recommended when the view is small with many pixels
>>> psf.plot2d(axlim=psf.support / 2,
               clim=(1e-9,1e0,
               interp_method='bicubic',
               fig=fig, ax=ax)
>>> plt.grid(False)
