***************
System Modeling
***************

In this example we will see how to model an end-to-end optical system using prysm.  Our system will have both an objective lens or telescope as well as a sensor with an optical low-pass filter.  We begin by importing the relevant classes, numpy, and setting some visual styles::

    import numpy as np
    from prysm import FringeZernike, PSF, MTF, PixelAperture, OLPF
    from matplotlib import pyplot as plt
    plt.style.use('bmh')

Next we model the PSF of the objective, given its aperture, focal length, and aberration coefficients::

    # data from a wavefront sensor, optical design program, etc...
    zernikes = [0, 0, 0, 0, 0.125, 0.125, -0.05, -0.05, 0.2]

    # a pentagonal aperture inscribed in a square 10mm on a side with 50mm EFL
    pupil = FringeZernike(zernikes, epd=10, mask='pentagon', opd_unit='um', norm=True)
    psf = PSF.from_pupil(pupil, efl=50)

Here we have implicitly accepted the default wavelength of 0.5 microns, and Q factor of 2 (Nyquist sampling) which are usually sane defaults.  The pupil is pentagonal in shape and is sufficiently described by a Zernike expansion up to Z9.

We can plot the PSF of the objective::

    psf.plot2d(axlim=25)
    plt.grid(False)

or compute its MTF::

    mtf = MTF.from_psf(psf)
    mtf.plot_tan_sag(max_freq=100)

But we are more concerned with the system-level performance::

    pixel_pitch = 5  # 5 micron pixels
    aa_filter = OLPF(pixel_pitch*0.66)
    pixel = PixelAperture(pixel_pitch)
    sys_psf = psf.conv(aa_filter).conv(pixel)

We can plot the system PSF, which is fairly abstract since it includes the pixel aperture::

    sys_psf.plot2d(axlim=25)
    plt.grid(False)

Of more interest is the system-level MTF::

    sys_mtf = MTF.from_psf(sys_psf)
    sys_mtf.plot_tan_sag(max_freq=100)


For more information on the classes used, see :doc:`Zernikes <../user_guide/zernikes>`, :doc:`PSFs <../user_guide/psfs>`, :doc:`MTFs <../user_guide/mtfs>`, and :doc:`PixelApertures, OLPFs, and convolutions <../user_guide/convolvables>`.
