Scalar Diffraction Propagation
===============================

prysm utilizes scalar diffraction theory to propogate pupils of optical systems to focal planes.  We note that the far-field [1]_ diffraction pattern of an object is its fourier transform.  We also note that the pupil of a system is a spatial modulator, manipulating both the ampltude and phase of a wave passing through it.  We also note that the role of a lens is to project an image of an object at infinity to its focal plane, and, consequently, if the far-field constraint is met we can represent a system's PSF as the fourier transform of its entrance pupil.  The EP is modeled as some function:

.. math::
    P(\rho,\theta) = A(\rho,\theta)\exp{(i \tfrac{2\pi}{\lambda} \phi(\rho,\theta))}

where :math:`\rho` is the radial pupil coordinate, in the range :math:`(0,1)`, :math:`\theta` is the azimuthal pupil coordinate, in the range :math:`(0,2\pi)`, :math:`\lambda` is the propagation wavelength, and :math:`\phi` is some phase modulation function.  Examples of :math:`\phi` include:

.. math::
    \rho^2 & \quad \text{defocus} \\
    \rho^4 & \quad \text{spherical aberration} \\
    \rho^3\cos{(\theta)} & \quad \text{coma} \\
    & \quad \text{etc.}

prysm utilizes numpy's fast fourier transform (FFT) routine to handle the computation.  The units in the two planes are linked by:

.. math::
    \xi_{\text{image}} = \frac{\lambda f}{\epsilon_{\text{pupil}} N}

where :math:`\xi` is the image plane sample spacing, :math:`f` is the focal length of the lens, :math:`\epsilon` is the pupil plane sample spacing, and :math:`N` is the number of samples.  Both planes must contain an equal number of samples.

If the pupil plane is not padded, the PSF plane will contain severe aliasing from the discrete FFT and not be representative of the true PSF.  prysm allows the specification of the padding in the :meth:`~prysm.PSF.from_pupil` method to the user's desire.  The default value of 1 ensures there will be no aliasing.  Increasing the padding will decrease the spacing between samples in the image plane.  Increasing the number of samples in the unpadded pupil will increase the width of the domain of the PSF.

.. [1] far-field implies that the Fresnel number of the source is < 1.  The fresnel number, :math:`F = \frac{a^2}{L\lambda}` where :math:`a` is the source radius, :math:`L` the observation distance, and :math:`\lambda` is the wavelength of light.