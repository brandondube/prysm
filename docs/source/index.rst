prysm
=====
:Release: |release|
:Date: |today|

prysm is an open-source library for physical and first-order modeling of optical systems and analysis of related data.  It is an unaffiliated sister library to PROPER and POPPY, codes developed for physical optics modeling of primarily space-based systems.  Prysm has a more restrictive capability in that domain, notably lacking multi-plane diffraction propagation, but also offers a broader set of features.

Use Cases
---------
prysm aims to be a swiss army knife for an optical engineer or student.  Its primary use cases include:

* Analysis of optical data
* robust numerical modeling of optical and opto-electronic systems based on physical optics
* wavefront sensing

Please see the Features section for more details.

Installation
-----------
prysm is available from either PyPi:

>>> pip install prysm

or github:

>>> pip install git+git://github.com/brandondube/prysm.git

It requires a minimal set of dependencies for scientific python; namely `numpy <http://www.numpy.org/>`_, `scipy <https://www.scipy.org/>`_, and `matplotlib <https://matplotlib.org/>`_.  It optionally depends on `numba <https://numba.pydata.org/>`_ (for acceleration of some routines on CPUs), `cupy <https://cupy.chainer.org/>`_ (for experimental use with nVidia GPUs), `imageio <https://imageio.github.io/>`_ (for reading and writing images), `h5py <https://www.h5py.org/>`_ (for reading of Zygo's datx format), and `pandas <https://pandas.pydata.org/>`_ (for some advanced utility MTF functions.  Pip can be instructed to install these alongside prysm,

>>> pip install prysm[cpu+]  // for numba

>>> pip install prysm[cuda] // for cupy

>>> pip install prysm[img] // for imageio

>>> pip install prysm[Mx]  // for h5py

>>> pip install prysm[mtf+] // for pandas

Features
--------

Physical Optics
~~~~~~~~~~~~~~~

* Modeing of pupil planes via or with:
* * Hopkins' wave aberration expansion, "Seidel aberrations"
* * Fringe Zernike polynomials up to Z48, unit amplitude or RMS
* * Zemax Standard Zernike polynomials up to Z48, unit amplitude
* * apodization
* * masks
* * * circles and ellipses
* * * n sided regular polygons
* * * user-provided
* * * synthetic interferograms
* * * PV and rms evaluation
* * * plotting

* Propagation of pupil planes via Fresnel transforms to Point Spread Functions (PSFs), which support
* * calculation and plotting of encircled energy
* * evaluation and plotting of slices
* * 2D plotting with or without power law scaling

* Computation of MTF from PSFs via the FFT method
* * MTF Full-Field Displays
* * MTF vs Field vs Focus
* * * Best Individual Focus
* * * Best Average Focus
* * evaluation at exact cartesian or polar spatial frequencies
* * 2D and slice plotting

* Rich tools for convolution of PSFs with images or synthetic objects:
* * pinholes
* * slits
* * Siemens stars
* * tilted squares
* display and reading of images

* Detector models for e.g. STOP analysis or image synthesis

* Interferometric analysis
* * cropping, masking
* * least-squares fitting and subtraction of Zernike modes, planes, and spheres
* * band-reject filters
* * evaluation of PV, RMS, Sa
* * computation of 2D PSD
* * plotting

First-Order Optics
~~~~~~~~~~~~~~~~~~
* object-image distance relation
* F/#, NA
* lateral and longitudinal magnification
* defocus-deltaZ relation
* two lens EFL and BFL

Parsing Data from Commercial Instruments
~~~~~~~~~~~~~~~~~~~~~
* Trioptics ImageMaster MTF benches
* Zygo Fizeau and white light interferometers


User's Guide
------------


Developer Guide
---------------

prysm's development has been a one-man affair for some number of years.  Contributions are appreciated in earnest.  These may take the form of e.g. improvements to documentation or docstrings, new unit tests to expand coverage and depth of testing, or development of new or expanded features.  Please contact the primary author to begin contributing, or file a PR/issue on github.
