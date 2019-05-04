*****
prysm
*****
:Release: |release|
:Date: |today|

prysm is an open-source library for physical and first-order modeling of optical systems and analysis of related data.  It is an unaffiliated sister library to PROPER and POPPY, codes developed to do physical optics modeling for primarily space-based systems.  Prysm has a more restrictive capability in that domain, notably lacking multi-plane diffraction propagation, but also offers a broader set of features.

.. contents::

Use Cases
---------
prysm aims to be a swiss army knife for optical engineers and students.  Its primary use cases include:

* Analysis of optical data
* robust numerical modeling of optical and opto-electronic systems based on physical optics
* wavefront sensing

Please see the Features section for more details.

prysm is on pypi:

>>> pip install prysm

prysm requires only `numpy <http://www.numpy.org/>`_ and `scipy <https://www.scipy.org/>`_.

If your environment has `numba <http://numba.pydata.org/>`_ installed, it will automatically accelerate many of prysm's compuations.  To use an nVidia GPU, you must have `cupy <https://cupy.chainer.org/>`_ installed.  Plotting uses `matplotlib <https://matplotlib.org/>`_.  Images are read and written with `imageio <https://imageio.github.io/>`_.  Some MTF utilities utilize `pandas <https://pandas.pydata.org/>`_.  Reading of Zygo datx files requires `h5py <https://www.h5py.org/>`_.

pip can be directed to install these,

>>> pip install prysm[cpu]     # for numba
>>> pip install prysm[cuda]    # for cupy
>>> pip install prysm[img]     # for imageio
>>> pip install prysm[Mx]      # for h5py
>>> pip install prysm[mtf]     # for pandas
>>> pip install prysm[deluxe]  # I want it all

or they may be installed at any time.

Features
--------

Physical Optics
~~~~~~~~~~~~~~~

* Modeing of pupil planes via or with:
* * Fringe Zernike polynomials up to Z48, unit amplitude or RMS
* * Noll ("Zemax Standard") Zernike polynomials up to Z36, unit amplitude or RMS
* * apodization
* * masks
* * * circles and ellipses
* * * n sided regular polygons
* * * user-provided
* * synthetic fringe maps
* * PV, RMS, stdev, Sa, Strehl evaluation
* * plotting

* Propagation of pupil planes via Fresnel transforms to Point Spread Functions (PSFs), which support
* * calculation and plotting of encircled energy
* * evaluation and plotting of slices
* * 2D plotting with or without power law or logarithmic scaling

* Computation of MTF from PSFs via the FFT method
* * MTF Full-Field Displays
* * MTF vs Field vs Focus
* * * Best Individual Focus
* * * Best Average Focus
* * evaluation at
* * * exact Cartesian spatial frequencies
* * * exact polar spatial frequencies
* * * Azimuthal average
* * 2D and slice plotting

* Rich tools for convolution of PSFs with images or synthetic objects:
* * pinholes
* * slits
* * Siemens stars
* * tilted squares
* * slanted edges
* * gratings
* * arrays of gratings
* * chirps
* read, write, and display of images

* Detector models for e.g. STOP analysis or image synthesis

* image-chain degredation models:
* * smear
* * jitter
* * atmospheric seeing

* Interferometric analysis
* * cropping
* * masking
* * spatial filtering
* * least-squares fitting and subtraction of Zernike modes, planes, and spheres
* * evaluation of PV, RMS, stdev, Sa, band-limited RMS, total integrated scatter
* * computation of PSD
* * * 2D
* * * x, y, azimuthally averaged slices
* * * evaluation and/or comparison to ab (power law) or abc (Lorentzian) models
* * spike clipping
* * plotting

First-Order Optics
~~~~~~~~~~~~~~~~~~
* object-image distance relations
* F/#, NA
* lateral and longitudinal magnification
* defocus-deltaZ relation
* two lens EFL and BFL

Parsing Data from Commercial & Open Source Instruments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Trioptics ImageMaster MTF benches
* Zygo interferometers
* SigFit
* MTF Mapper


User's Guide
------------

.. toctree::

   user_guide/index.rst


Examples
--------

.. toctree::

    examples/index.rst


API Reference
-------------

.. toctree::

    api/index.rst

Contributing
------------

.. toctree::

    contributing.rst

Release History
---------------

.. toctree::

    releases/index.rst
